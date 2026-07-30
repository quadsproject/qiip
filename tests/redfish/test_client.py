"""Unit tests for RedfishClient power state, power actions, and error mapping."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError

BMC_TEMPLATE = "mgmt-{hostname}"
SYSTEM_ID = "1"
SYSTEMS_URL = "https://mgmt-server01/redfish/v1/Systems/1"
RESET_URL = "https://mgmt-server01/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"


class TestGetPowerState:
    async def test_returns_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            state = await rc.get_power_state("server01")
        assert state == "On"

    async def test_returns_off(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            state = await rc.get_power_state("server01")
        assert state == "Off"

    async def test_returns_powering_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "PoweringOn"})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            state = await rc.get_power_state("server01")
        assert state == "PoweringOn"

    async def test_raises_on_http_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, status_code=500, json={})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            with pytest.raises(RedfishError):
                await rc.get_power_state("server01")

    async def test_raises_on_connect_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"), url=SYSTEMS_URL
        )
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            with pytest.raises(RedfishError):
                await rc.get_power_state("server01")


class TestPowerActionIdempotent:
    async def test_skip_when_already_on(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "On"})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            state = await rc.power_action("server01", "On")
        assert state == "On"
        assert len(httpx_mock.get_requests()) == 1  # Only GET, no POST

    async def test_skip_when_already_off(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=SYSTEMS_URL, json={"PowerState": "Off"})
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
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
            rc = RedfishClient(
                client, BMC_TEMPLATE, SYSTEM_ID, poll_timeout=5.0, poll_interval=0.01
            )
            state = await rc.power_action("server01", "ForceOff")

        assert state == "Off"
        requests = httpx_mock.get_requests()
        post_reqs = [r for r in requests if r.method == "POST"]
        assert len(post_reqs) == 1
        import json

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
            rc = RedfishClient(
                client, BMC_TEMPLATE, SYSTEM_ID, poll_timeout=5.0, poll_interval=0.01
            )
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
            rc = RedfishClient(
                client, BMC_TEMPLATE, SYSTEM_ID, poll_timeout=0.1, poll_interval=0.02
            )
            with pytest.raises(RedfishError, match="Off"):
                await rc.power_action("server01", "ForceOff")


class TestResolveBmcHost:
    async def test_template_substitution(self) -> None:
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, "bmc-{hostname}.lab", SYSTEM_ID)
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
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            with pytest.raises(RedfishError) as exc_info:
                await rc.get_power_state("server01")
        assert exc_info.value.human_message == "This action is not supported by the BMC"

    async def test_invalid_action_raises(self, httpx_mock: HTTPXMock) -> None:
        async with httpx.AsyncClient() as client:
            rc = RedfishClient(client, BMC_TEMPLATE, SYSTEM_ID)
            with pytest.raises(RedfishError, match="Unsupported"):
                await rc.power_action("server01", "BogusAction")
