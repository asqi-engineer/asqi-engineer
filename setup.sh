#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# GitHub repository information
REPO="asqi-engineer/asqi-engineer"
BRANCH="main"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        print_error "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed or not in PATH"
        exit 1
    fi
    
}

# Directory structure will be created automatically when downloading config files

# Download file with error handling
download_file() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    
    print_status "Downloading ${description}..."
    
    if curl -fsSL "$url" -o "$output_path"; then
        print_success "Downloaded ${description} to ${output_path}"
    else
        print_error "Failed to download ${description} from ${url}"
        exit 1
    fi
}

# Download config folder structure
download_config_folder() {
    print_status "Downloading configuration files..."
    
    # Download all configuration files
    config_files=(
        "config/docker_container.yaml"
        "config/score_cards/asqi_chatbot_score_card.yaml"
        "config/score_cards/example_score_card.yaml"
        "config/score_cards/garak_score_card.yaml"
        "config/suites/asqi_chatbot_test_suite.yaml"
        "config/suites/chatbot_simulator_test.yaml"
        "config/suites/deepteam_test.yaml"
        "config/suites/demo_test.yaml"
        "config/suites/garak_test.yaml"
        "config/suites/trustllm_test.yaml"
        "config/systems/demo_systems.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        # Create directory if it doesn't exist
        config_dir=$(dirname "$config_file")
        mkdir -p "$config_dir"
        
        # Download the file
        download_file "${BASE_URL}/${config_file}" "$config_file" "$(basename $config_file)"
    done
}

# Download required files
download_files() {
    print_status "Downloading required files from GitHub repository..."
    
    # Download docker-compose.yml from docker directory
    download_file "${BASE_URL}/docker/docker-compose.yml" "docker-compose.yml" "Docker Compose configuration"
    
    # Download litellm config
    download_file "${BASE_URL}/litellm_config.yaml" "litellm_config.yaml" "LiteLLM configuration"
    
    # Download all config files
    download_config_folder
}

# Create .env file
create_env_file() {
    print_status "Creating .env file..."
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists. Creating backup as .env.backup"
        cp .env .env.backup
    fi
    
    cat > .env << 'EOF'
# LLM API Keys
LITELLM_MASTER_KEY="sk-1234"
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AWS_BEARER_TOKEN_BEDROCK=

# Otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces

# DB
DBOS_DATABASE_URL=postgres://postgres:asqi@localhost:5432/asqi_starter
# # Docker host url might need to be configured if not at default location (e.g. if using macOS)
# # If so, uncomment below. The default location for macOS is shown below but you might need to change it.
# DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
EOF
    
    print_success "Created .env file"
}

# Main setup function
main() {
    echo ""
    echo "🚀 ASQI Engineer Quick Setup"
    echo "================================"
    echo ""
    
    check_prerequisites
    download_files
    create_env_file
    
    echo ""
    print_success "Setup completed successfully! 🎉"
    echo ""
    echo "Downloaded files:"
    echo "- docker-compose.yml (includes PostgreSQL database, LiteLLM proxy, and Jaeger tracing)"
    echo "- litellm_config.yaml (LLM provider configuration)"
    echo "- config/ folder with test suites, systems, and score cards for:"
    echo "  • Demo testing (mock tester)"
    echo "  • Security testing (Garak)"
    echo "  • Red team testing (DeepTeam)"
    echo "  • Trust evaluation (TrustLLM)"
    echo "  • Chatbot simulation and evaluation"
    echo ""
    echo "Next steps:"
    echo "1. Configure your API keys in the .env file:"
    echo "   - Add your OPENAI_API_KEY, ANTHROPIC_API_KEY, or AWS_BEARER_TOKEN_BEDROCK"
    echo "   - Keep LITELLM_MASTER_KEY as 'sk-1234' for the demo"
    echo ""
    echo "2. Start the services:"
    echo "   docker compose up -d"
    echo ""
    echo "3. Run your first test:"
    echo "   asqi execute-tests -t config/suites/demo_test.yaml -s config/systems/demo_systems.yaml"
    echo ""
    echo "4. Explore other test suites:"
    echo "   ls config/suites/  # See all available test configurations"
    echo ""
    echo "For more information, visit: https://www.asqi.ai/quickstart.html"
    echo ""
}

# Run main function
main "$@"
