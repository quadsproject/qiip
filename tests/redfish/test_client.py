"""Unit tests for RedfishClient power state, power actions, and error mapping."""

from __future__ import annotations

import asyncio
import inspect
import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

import inference_proxy.redfish.client as redfish_module
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError

try:
    from inference_proxy.redfish.errors import RedfishDestinationError
except ImportError:  # Before-fix compatibility for behavioral comparison.
    RedfishDestinationError = RedfishError

BMC_TEMPLATE = "mgmt-{hostname}"
SYSTEM_ID = "1"
SYSTEMS_URL = "https://mgmt-server01/redfish/v1/Systems/1"
RESET_URL = "https://mgmt-server01/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"


def _policy(*hosts: str) -> EndpointPolicy:
    return EndpointPolicy.from_values(
        allowed_hosts=hosts or ["server01"],
        allowed_networks=["10.0.0.0/8"],
        allowed_ports=[8000],
    )


def _redfish_client(
    client: httpx.AsyncClient,
    template: str = BMC_TEMPLATE,
    *,
    allowed_hosts: tuple[str, ...] = ("server01",),
    auth: httpx.Auth | None = None,
    poll_timeout: float = 60.0,
    poll_interval: float = 5.0,
) -> RedfishClient:
    """Build against both old main and the destination-safe constructor.

    The compatibility branch lets S1 fail on its outbound request against
    unfixed main instead of failing mechanically on a changed signature.
    """
    resolved_auth = auth or httpx.BasicAuth("admin", "secret")
    if "hostname_policy" not in inspect.signature(RedfishClient).parameters:
        client.auth = resolved_auth
        return RedfishClient(
            client,
            template,
            SYSTEM_ID,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
    return RedfishClient(
        client,
        template,
        SYSTEM_ID,
        hostname_policy=_policy(*allowed_hosts),
        auth=resolved_auth,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )


class TestGetPowerState:
    async def test_returns_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            state = await rc.get_power_state("server01")
        assert state == "On"

    async def test_returns_off(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            state = await rc.get_power_state("server01")
        assert state == "Off"

    async def test_returns_powering_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "PoweringOn"})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            state = await rc.get_power_state("server01")
        assert state == "PoweringOn"

    async def test_raises_on_http_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, status_code=500, json={})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            with pytest.raises(RedfishError):
                await rc.get_power_state("server01")

    async def test_raises_on_connect_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"), url=SYSTEMS_URL
        )
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            with pytest.raises(RedfishError):
                await rc.get_power_state("server01")


class TestPowerActionIdempotent:
    async def test_skip_when_already_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            state = await rc.power_action("server01", "On")
        assert state == "On"
        assert len(httpx_mock.get_requests()) == 1  # Only GET, no POST

    async def test_skip_when_already_off(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            state = await rc.power_action("server01", "ForceOff")
        assert state == "Off"
        assert len(httpx_mock.get_requests()) == 1


class TestPowerAction:
    async def test_force_off_posts_reset_and_polls(self, httpx_mock: HTTPXMock) -> None:
        # GET: current state is On
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        # POST: reset accepted
        httpx_mock.add_response(url=RESET_URL, status_code=200)
        # GET (poll): now Off
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})

        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client, poll_timeout=5.0, poll_interval=0.01)
            state = await rc.power_action("server01", "ForceOff")

        assert state == "Off"
        requests = httpx_mock.get_requests()
        post_reqs = [r for r in requests if r.method == "POST"]
        assert len(post_reqs) == 1
        body = json.loads(post_reqs[0].content)
        assert body["ResetType"] == "ForceOff"

    async def test_on_posts_reset_and_polls(self, httpx_mock: HTTPXMock) -> None:
        # GET: current state is Off
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})
        # POST: reset accepted
        httpx_mock.add_response(url=RESET_URL, status_code=200)
        # GET (poll): now On
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})

        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client, poll_timeout=5.0, poll_interval=0.01)
            state = await rc.power_action("server01", "On")

        assert state == "On"


class TestPowerActionTimeout:
    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
    async def test_raises_on_poll_timeout(self, httpx_mock: HTTPXMock) -> None:
        # GET: current state is On (want Off via ForceOff)
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        # POST: reset accepted
        httpx_mock.add_response(url=RESET_URL, status_code=200)

        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client, poll_timeout=0.1, poll_interval=0.02)
            with pytest.raises(RedfishError, match="Off"):
                await rc.power_action("server01", "ForceOff")


class TestResolveBmcHost:
    async def test_template_substitution(self) -> None:
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(
                client,
                "bmc-{hostname}.lab",
                allowed_hosts=("node42",),
            )
            assert rc._resolve_bmc_host("node42") == "bmc-node42.lab"


class TestErrorMapping:
    async def test_known_message_id_mapped(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=SYSTEMS_URL,
            status_code=400,
            json={
                "error": {
                    "@Message.ExtendedInfo": [
                        {
                            "MessageId": "Base.1.12.ActionNotSupported",
                            "Message": "The action is not supported.",
                        }
                    ]
                }
            },
        )
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            with pytest.raises(RedfishError) as exc_info:
                await rc.get_power_state("server01")
        assert exc_info.value.human_message == "This action is not supported by the BMC"

    async def test_invalid_action_raises(self, httpx_mock: HTTPXMock) -> None:
        async with httpx.AsyncClient() as client:
            rc = _redfish_client(client)
            with pytest.raises(RedfishError, match="Unsupported"):
                await rc.power_action("server01", "BogusAction")


@pytest.mark.parametrize("action", ["GracefulRestart", "ForceRestart"])
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_restart_posts_reset_when_already_on(
    httpx_mock: HTTPXMock,
    action: str,
) -> None:
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
    httpx_mock.add_response(url=RESET_URL, status_code=200)
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})

    async with httpx.AsyncClient() as client:
        state = await _redfish_client(client).power_action("server01", action)

    assert state == "On"
    post_requests = [
        request for request in httpx_mock.get_requests() if request.method == "POST"
    ]
    assert len(post_requests) == 1
    assert json.loads(post_requests[0].content) == {"ResetType": action}


@pytest.mark.parametrize(
    ("action", "transition", "target"),
    [
        ("On", "PoweringOn", "On"),
        ("ForceOff", "PoweringOff", "Off"),
    ],
)
async def test_matching_transition_polls_without_reset(
    httpx_mock: HTTPXMock,
    action: str,
    transition: str,
    target: str,
) -> None:
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": transition})
    httpx_mock.add_response(url=RESET_URL, status_code=200, is_optional=True)
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": target})

    async with httpx.AsyncClient() as client:
        state = await _redfish_client(client).power_action("server01", action)

    assert state == target
    assert [request.method for request in httpx_mock.get_requests()] == ["GET", "GET"]


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += delay


async def test_poll_tolerates_transient_error_after_reset(
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(redfish_module, "_monotonic", clock.monotonic)
    monkeypatch.setattr(redfish_module, "_sleep", clock.sleep)
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
    httpx_mock.add_response(url=RESET_URL, status_code=200)
    httpx_mock.add_response(url=SYSTEMS_URL, status_code=500, json={})
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})

    async with httpx.AsyncClient() as client:
        action = _redfish_client(client, poll_timeout=3.0, poll_interval=1.0)
        state = await asyncio.wait_for(
            action.power_action("server01", "ForceOff"),
            timeout=1.0,
        )

    assert state == "Off"
    assert (
        len(
            [
                request
                for request in httpx_mock.get_requests()
                if request.method == "POST"
            ]
        )
        == 1
    )


async def test_poll_performs_final_deadline_probe(
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(redfish_module, "_monotonic", clock.monotonic)
    monkeypatch.setattr(redfish_module, "_sleep", clock.sleep)
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
    httpx_mock.add_response(url=RESET_URL, status_code=200)
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})

    async with httpx.AsyncClient() as client:
        action = _redfish_client(client, poll_timeout=1.0, poll_interval=1.0)
        state = await asyncio.wait_for(
            action.power_action("server01", "ForceOff"),
            timeout=1.0,
        )

    assert state == "Off"
    assert clock.value == 1.0
    assert len(httpx_mock.get_requests()) == 4


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{}", "missing PowerState"),
        (b"<html>error</html>", "malformed JSON"),
        (b'{"PowerState": null}', "None"),
        (b'{"PowerState": 7}', "7"),
        (b'{"PowerState": "Paused"}', "Paused"),
        (b"[]", "missing PowerState"),
    ],
)
async def test_malformed_power_state_is_redfish_error(
    httpx_mock: HTTPXMock,
    content: bytes,
    message: str,
) -> None:
    httpx_mock.add_response(url=SYSTEMS_URL, content=content)

    async with httpx.AsyncClient() as client:
        with pytest.raises(RedfishError, match=message):
            await _redfish_client(client).get_power_state("server01")


async def test_programming_error_not_wrapped(
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})

    def broken_json(_response: httpx.Response) -> object:
        raise AttributeError("programming defect")

    monkeypatch.setattr(httpx.Response, "json", broken_json)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AttributeError, match="programming defect"):
            await _redfish_client(client).get_power_state("server01")


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_unapproved_bmc_destination_sends_no_credentials(
    httpx_mock: HTTPXMock,
) -> None:
    attacker_url = "https://mgmt-attacker.example.net/redfish/v1/Systems/1"
    httpx_mock.add_response(url=attacker_url, json={"PowerState": "On"})

    async with httpx.AsyncClient() as client:
        redfish = _redfish_client(client)
        with pytest.raises(RedfishDestinationError, match="not allowed"):
            await redfish.get_power_state("attacker.example.net")

    assert httpx_mock.get_requests() == []


async def test_approved_bmc_destination_sends_credentials(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
    auth = httpx.BasicAuth("operator", "redfish-secret")

    async with httpx.AsyncClient() as client:
        state = await _redfish_client(client, auth=auth).get_power_state("SERVER01")

    assert state == "On"
    [request] = httpx_mock.get_requests()
    assert str(request.url) == SYSTEMS_URL
    assert request.headers["Authorization"] == "Basic b3BlcmF0b3I6cmVkZmlzaC1zZWNyZXQ="
