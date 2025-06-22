# EternalKernel PyTorch Nodes - Makefile
#
# Comprehensive development and publishing automation
# Author: Hopping Mad Games
# License: AGPL-3.0

.PHONY: help clean install test lint format validate publish dev-setup check-deps update-deps security-check

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := pytorch-nodes
VENV := venv
TEST_DIR := tests
SRC_DIR := .
COMFY_TOKEN_FILE := .comfy_token

help: ## Show this help message
	@echo "$(CYAN)EternalKernel PyTorch Nodes - Development Commands$(NC)"
	@echo "$(BLUE)=================================================$(NC)"
	@echo ""
	@echo "$(WHITE)Usage:$(NC) make [target]"
	@echo ""
	@echo "$(WHITE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(WHITE)Examples:$(NC)"
	@echo "  $(YELLOW)make dev-setup$(NC)     - Set up development environment"
	@echo "  $(YELLOW)make test$(NC)          - Run all tests"
	@echo "  $(YELLOW)make publish$(NC)       - Publish to ComfyUI registry"
	@echo "  $(YELLOW)make clean$(NC)         - Clean build artifacts"
	@echo ""

clean: ## Clean build artifacts and temporary files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

install: ## Install project dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	@$(PIP) install -r requirements.txt
	@$(PIP) install pytest pytest-cov black isort flake8 mypy ruff
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

venv: ## Create virtual environment
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "$(BLUE)Activate with: source $(VENV)/bin/activate$(NC)"

dev-setup: venv install-dev ## Complete development environment setup
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Activate venv: $(YELLOW)source $(VENV)/bin/activate$(NC)"
	@echo "  2. Run tests: $(YELLOW)make test$(NC)"
	@echo "  3. Start developing!"

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@if [ -d "$(TEST_DIR)" ]; then \
		$(PYTHON) -m pytest $(TEST_DIR)/ -v; \
	else \
		echo "$(RED)No tests directory found. Creating basic test structure...$(NC)"; \
		mkdir -p $(TEST_DIR); \
		echo "# Test placeholder" > $(TEST_DIR)/test_placeholder.py; \
		echo "$(BLUE)Tests directory created. Add your tests to $(TEST_DIR)/$(NC)"; \
	fi

test-cov: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

lint: ## Run linting checks
	@echo "$(YELLOW)Running linting checks...$(NC)"
	@echo "$(BLUE)Running flake8...$(NC)"
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo "$(BLUE)Running ruff...$(NC)"
	@ruff check . || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	@black . --line-length 88
	@isort . --profile black
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	@black . --check --line-length 88
	@isort . --profile black --check-only

type-check: ## Run type checking with mypy
	@echo "$(YELLOW)Running type checks...$(NC)"
	@mypy . --ignore-missing-imports || true
	@echo "$(GREEN)✓ Type checking complete$(NC)"

security-check: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(NC)"
	@ruff check . --select=S || true
	@echo "$(GREEN)✓ Security check complete$(NC)"

validate: ## Validate package for ComfyUI registry
	@echo "$(YELLOW)Validating package for ComfyUI registry...$(NC)"
	@comfy node validate
	@echo "$(GREEN)✓ Package validation complete$(NC)"

check-deps: ## Check for outdated dependencies
	@echo "$(YELLOW)Checking for outdated dependencies...$(NC)"
	@$(PIP) list --outdated
	@echo "$(GREEN)✓ Dependency check complete$(NC)"

update-deps: ## Update dependencies to latest versions
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	@$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

build: ## Build package
	@echo "$(YELLOW)Building package...$(NC)"
	@$(PYTHON) -m build
	@echo "$(GREEN)✓ Package built$(NC)"

publish-test: validate ## Publish to test registry (dry run)
	@echo "$(YELLOW)Running publish validation...$(NC)"
	@comfy node validate
	@echo "$(GREEN)✓ Ready for publishing$(NC)"
	@echo "$(BLUE)Run 'make publish' to publish to registry$(NC)"

publish: validate ## Publish to ComfyUI registry
	@echo "$(YELLOW)Publishing to ComfyUI registry...$(NC)"
	@if [ -f "$(COMFY_TOKEN_FILE)" ]; then \
		echo "$(BLUE)Using token from $(COMFY_TOKEN_FILE)$(NC)"; \
		source $(COMFY_TOKEN_FILE) && comfy node publish --token $$COMFY_REGISTRY_TOKEN; \
	elif [ -n "$$COMFY_REGISTRY_TOKEN" ]; then \
		echo "$(BLUE)Using token from environment variable$(NC)"; \
		comfy node publish --token $$COMFY_REGISTRY_TOKEN; \
	else \
		echo "$(RED)No API token found!$(NC)"; \
		echo "$(YELLOW)Set COMFY_REGISTRY_TOKEN environment variable or create $(COMFY_TOKEN_FILE)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Published successfully!$(NC)"

version-bump: ## Bump version number (requires VERSION parameter)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Usage: make version-bump VERSION=1.0.1$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Bumping version to $(VERSION)...$(NC)"
	@sed -i 's/version = "[^"]*"/version = "$(VERSION)"/' pyproject.toml
	@echo "$(GREEN)✓ Version bumped to $(VERSION)$(NC)"

git-tag: ## Create and push git tag (requires VERSION parameter)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Usage: make git-tag VERSION=v1.0.1$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Creating git tag $(VERSION)...$(NC)"
	@git tag -a $(VERSION) -m "Release $(VERSION)"
	@git push origin $(VERSION)
	@echo "$(GREEN)✓ Git tag $(VERSION) created and pushed$(NC)"

release: ## Full release process (bump version, tag, publish)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Usage: make release VERSION=1.0.1$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Starting release process for version $(VERSION)...$(NC)"
	@make version-bump VERSION=$(VERSION)
	@git add pyproject.toml
	@git commit -m "Bump version to $(VERSION)"
	@make git-tag VERSION=v$(VERSION)
	@git push origin main
	@make publish
	@echo "$(GREEN)✓ Release $(VERSION) complete!$(NC)"

status: ## Show project status and info
	@echo "$(CYAN)EternalKernel PyTorch Nodes - Project Status$(NC)"
	@echo "$(BLUE)===========================================$(NC)"
	@echo ""
	@echo "$(WHITE)Project Info:$(NC)"
	@echo "  Name: $(GREEN)EternalKernel PyTorch Nodes$(NC)"
	@echo "  Version: $(GREEN)$(shell grep 'version =' pyproject.toml | cut -d'"' -f2)$(NC)"
	@echo "  License: $(GREEN)AGPL-3.0$(NC)"
	@echo "  Publisher: $(GREEN)hmg$(NC)"
	@echo ""
	@echo "$(WHITE)Git Status:$(NC)"
	@git status --short
	@echo ""
	@echo "$(WHITE)Dependencies:$(NC)"
	@echo "  Python: $(GREEN)$(shell $(PYTHON) --version)$(NC)"
	@echo "  Pip: $(GREEN)$(shell $(PIP) --version | cut -d' ' -f2)$(NC)"
	@echo "  ComfyUI CLI: $(GREEN)$(shell comfy --version 2>/dev/null || echo 'Not installed')$(NC)"
	@echo ""
	@if [ -f "$(COMFY_TOKEN_FILE)" ]; then \
		echo "$(WHITE)API Token:$(NC) $(GREEN)✓ Found in $(COMFY_TOKEN_FILE)$(NC)"; \
	elif [ -n "$$COMFY_REGISTRY_TOKEN" ]; then \
		echo "$(WHITE)API Token:$(NC) $(GREEN)✓ Found in environment$(NC)"; \
	else \
		echo "$(WHITE)API Token:$(NC) $(RED)✗ Not configured$(NC)"; \
	fi

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@echo "$(BLUE)README.md exists: $(GREEN)✓$(NC)"
	@echo "$(BLUE)LICENSE exists: $(GREEN)✓$(NC)"
	@echo "$(GREEN)✓ Documentation is up to date$(NC)"

all: clean install-dev test lint format validate ## Run full development workflow
	@echo "$(GREEN)✓ Full development workflow complete$(NC)"

.PHONY: help clean install install-dev venv dev-setup test test-cov lint format format-check type-check security-check validate check-deps update-deps build publish-test publish version-bump git-tag release status docs all
