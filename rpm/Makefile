PYTHON := python3
# Package name: qiip (stable train) or qiip-dev (development train), passed
# by the release workflow. Same template, one placeholder difference.
NAME ?= qiip

# python-semantic-release writes the version into pyproject.toml; dev train
# versions use the PEP 440 prerelease form 0.2.0-dev.1.
# The Makefile lives in rpm/; version is read from the repo-root pyproject.
GIT_VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('../pyproject.toml','rb'))['project']['version'])")
# RPM Version fields cannot contain hyphens.
VERSION := $(subst -,.,$(GIT_VERSION))
RELEASE := 1
DATE := $(shell date +'%a %b %d %Y')
TARBALL := v$(GIT_VERSION).tar.gz
CONFLICTS :=
ifeq ($(NAME),qiip-dev)
CONFLICTS := Conflicts: qiip
endif

ifeq ($(strip $(GIT_VERSION)),)
$(error Could not determine package version from pyproject.toml)
endif

all: $(TARBALL) $(NAME).spec

.PHONY: $(TARBALL)
$(TARBALL):
	rm -f $(TARBALL)
	# Same layout as the GitHub tag archive: prefix qiip-v<version>/.
	( cd .. && git archive --prefix qiip-v$(GIT_VERSION)/ HEAD ) | gzip > "$@"

$(NAME).spec: qiip.spec.tpl
	@echo "  GEN $@" ; \
	sed -e "s|@NAME@|$(NAME)|g" \
	    -e "s|@VERSION@|$(VERSION)|g" \
	    -e "s|@GIT_VERSION@|$(GIT_VERSION)|g" \
	    -e "s|@RELEASE@|$(RELEASE)|g" \
	    -e "s|@DATE@|$(DATE)|g" \
	    -e "s|@CONFLICTS@|$(CONFLICTS)|g" \
	    $< > $@.output ; \
	mv "$@.output" "$@" # move only on success

srpm: all
	dir=`pwd` ; \
	rm -f *.src.rpm ; \
	rpmbuild --define "_sourcedir $$dir" \
		 --define "_rpmdir    $$dir" \
		 --define "_builddir  $$dir" \
		 --define "_specdir   $$dir" \
		 --define "_srcrpmdir $$dir" \
	     -bs $(NAME).spec

rpmlint: all
	rm -rf $(CURDIR)/rpmbuild
	mkdir -p $(CURDIR)/rpmbuild/SOURCES $(CURDIR)/rpmbuild/SPECS
	cp $(TARBALL) $(CURDIR)/rpmbuild/SOURCES/
	cp $(NAME).spec $(CURDIR)/rpmbuild/SPECS/
	echo "==> Building RPM from spec"
	rpmbuild -ba --define "_topdir $(CURDIR)/rpmbuild" $(CURDIR)/rpmbuild/SPECS/$(NAME).spec
	echo "==> Running rpmlint"
	rpmlint $(CURDIR)/$(NAME).spec $$(find $(CURDIR)/rpmbuild/RPMS $(CURDIR)/rpmbuild/SRPMS -name '*.rpm' -type f 2>/dev/null)

version:
	@echo $(GIT_VERSION)

rpm-version:
	@echo $(VERSION)

clean:
	rm -rf *.src.rpm $(TARBALL) $(NAME).spec noarch rpmbuild
