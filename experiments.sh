#!/bin/bash

# Comprehensive experiments script for Fetch search algorithms
# Runs all search algorithms across all datasets with automatic git commits

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to update dataset configuration
update_dataset_config() {
    local dataset_name=$1
    local dataset_file=$2
    
    print_status "Updating dataset configuration to: $dataset_name ($dataset_file)"
    
    # Update the experiments_config.env file
    sed -i "s|^PATH_TO_DATASET=.*|PATH_TO_DATASET=$dataset_file|" search/experiments_config.env
    
    print_success "Dataset configuration updated to $dataset_name"
}

# Function to run a search algorithm
run_search_algorithm() {
    local algorithm=$1
    local dataset_name=$2
    local script_path=$3
    
    print_status "Running $algorithm on $dataset_name dataset"
    
    # Change to the appropriate directory
    cd "$(dirname "$script_path")"
    
    # Run the algorithm
    if python "$(basename "$script_path")"; then
        print_success "$algorithm completed successfully on $dataset_name"
        return 0
    else
        print_error "$algorithm failed on $dataset_name"
        return 1
    fi
}

# Function to commit and push results
commit_and_push() {
    local algorithm=$1
    local dataset_name=$2
    
    print_status "Committing and pushing results for $algorithm-$dataset_name"
    
    # Add all changes
    git add .
    
    # Commit with descriptive message
    git commit -m "experiment ran for $algorithm-$dataset_name" || {
        print_warning "Nothing to commit for $algorithm-$dataset_name"
        return 0
    }
    
    # Push to remote
    git push || {
        print_error "Failed to push changes for $algorithm-$dataset_name"
        return 1
    }
    
    print_success "Results committed and pushed for $algorithm-$dataset_name"
}

# Main experiment function
run_experiments() {
    print_status "Starting comprehensive experiments"
    print_status "Model configuration from server_config.env:"
    echo "POLICY_MODEL_PATH: $(grep POLICY_MODEL_PATH server_config.env | cut -d'=' -f2)"
    echo "VERIFIER_MODEL_PATH: $(grep VERIFIER_MODEL_PATH server_config.env | cut -d'=' -f2)"
    echo "EMBEDDING_MODEL_PATH: $(grep EMBEDDING_MODEL_PATH server_config.env | cut -d'=' -f2)"
    echo ""
    
    # Add dataset paths to experiments_config.env if they don't exist
    print_status "Setting up dataset paths in experiments_config.env"
    
    # Add missing dataset paths
    if ! grep -q "MATH500_PATH" search/experiments_config.env; then
        echo "MATH500_PATH = ../../dataset/math500/math500.jsonl" >> search/experiments_config.env
    fi
    
    if ! grep -q "AIME_PATH" search/experiments_config.env; then
        echo "AIME_PATH = ../../dataset/aime/aime.jsonl" >> search/experiments_config.env
    fi
    
    # Source the experiments_config.env to get the actual dataset paths
    print_status "Loading dataset paths from experiments_config.env"
    
    # Create a temporary file with proper bash syntax
    temp_env=$(mktemp)
    # Convert the env file format (KEY = VALUE) to bash format (KEY=VALUE)
    sed 's/ = /=/' search/experiments_config.env | grep -E '^[A-Z_]+_PATH=' > "$temp_env"
    
    # Source the temporary file to load the variables
    source "$temp_env"
    
    # Clean up temporary file
    rm "$temp_env"
    
    # Define datasets with actual paths from the environment
    declare -A datasets
    datasets["gsm8k"]="$GSM8K_TEST_PATH"
    datasets["math500"]="$MATH500_PATH"
    datasets["aime"]="$AIME_PATH"
    
    # Verify dataset paths exist
    print_status "Verifying dataset paths:"
    for dataset_name in "${!datasets[@]}"; do
        if [[ -f "${datasets[$dataset_name]}" ]]; then
            print_success "$dataset_name: ${datasets[$dataset_name]}"
        else
            print_error "$dataset_name: ${datasets[$dataset_name]} (FILE NOT FOUND)"
            exit 1
        fi
    done
    echo ""
    
    # Define search algorithms
    declare -A algorithms
    algorithms["beamsearch"]="search/beamsearch/beamsearch.py"
    algorithms["beamsearch_merge"]="search/beamsearch/beamsearch_merge.py"
    algorithms["bfs"]="search/bfs/bfs.py"
    algorithms["bfs_merge"]="search/bfs/bfs_merge.py"
    algorithms["run_mcts"]="search/mcts/run_mcts.py"
    algorithms["run_mcts_merge"]="search/mcts/run_mcts_merge.py"
    algorithms["ssdp"]="search/SSDP/SSDP.py"
    
    # Track total experiments
    total_experiments=$((${#datasets[@]} * ${#algorithms[@]}))
    current_experiment=0
    
    print_status "Total experiments to run: $total_experiments"
    print_status "Estimated time: $((total_experiments * 30)) minutes (assuming 30 min per experiment)"
    echo ""
    
    # Run experiments
    for dataset_name in "${!datasets[@]}"; do
        print_status "=== Starting experiments on $dataset_name dataset ==="
        
        # Update dataset configuration
        update_dataset_config "$dataset_name" "${datasets[$dataset_name]}"
        
        for algorithm in "${!algorithms[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            print_status "[$current_experiment/$total_experiments] Running $algorithm on $dataset_name"
            
            # Run the algorithm
            if run_search_algorithm "$algorithm" "$dataset_name" "${algorithms[$algorithm]}"; then
                # Commit and push results
                commit_and_push "$algorithm" "$dataset_name"
                print_success "Completed $algorithm-$dataset_name (experiment $current_experiment/$total_experiments)"
            else
                print_error "Failed $algorithm-$dataset_name (experiment $current_experiment/$total_experiments)"
                # Still try to commit any partial results
                commit_and_push "$algorithm" "$dataset_name"
            fi
            
            echo ""
            sleep 5  # Brief pause between experiments
        done
        
        print_success "Completed all experiments on $dataset_name dataset"
        echo "=========================================="
        echo ""
    done
    
    print_success "All experiments completed!"
    print_status "Summary:"
    print_status "- Datasets tested: ${!datasets[*]}"
    print_status "- Algorithms tested: ${!algorithms[*]}"
    print_status "- Total experiments: $total_experiments"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository. Please initialize git first."
        exit 1
    fi
    
    # Check if server_config.env exists
    if [[ ! -f "server_config.env" ]]; then
        print_error "server_config.env not found. Please ensure it exists."
        exit 1
    fi
    
    # Check if experiments_config.env exists
    if [[ ! -f "search/experiments_config.env" ]]; then
        print_error "search/experiments_config.env not found. Please ensure it exists."
        exit 1
    fi
    
    # Check if dataset directories exist
    for dataset in gsm8k math500 aime; do
        if [[ ! -d "dataset/$dataset" ]]; then
            print_error "Dataset directory dataset/$dataset not found."
            exit 1
        fi
    done
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please ensure Python is installed and in PATH."
        exit 1
    fi
    
    print_success "All prerequisites met!"
}

# Function to show help
show_help() {
    echo "Fetch Experiments Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --dry-run      Show what would be run without executing"
    echo "  --check        Check prerequisites and exit"
    echo ""
    echo "This script will:"
    echo "1. Run all search algorithms (beamsearch, beamsearch_merge, bfs, bfs_merge, run_mcts, run_mcts_merge, ssdp)"
    echo "2. Test on all datasets (gsm8k, math500, aime)"
    echo "3. Update dataset configuration between runs"
    echo "4. Commit and push results after each experiment"
    echo ""
    echo "Make sure your server is running before starting experiments!"
}

# Function for dry run
dry_run() {
    print_status "DRY RUN MODE - No experiments will be executed"
    echo ""
    
    # Add missing dataset paths first
    if ! grep -q "MATH500_PATH" search/experiments_config.env; then
        echo "MATH500_PATH = ../../dataset/math500/math500.jsonl" >> search/experiments_config.env
    fi
    
    if ! grep -q "AIME_PATH" search/experiments_config.env; then
        echo "AIME_PATH = ../../dataset/aime/aime.jsonl" >> search/experiments_config.env
    fi
    
    # Source the experiments_config.env to get the actual dataset paths
    temp_env=$(mktemp)
    sed 's/ = /=/' search/experiments_config.env | grep -E '^[A-Z_]+_PATH=' > "$temp_env"
    source "$temp_env"
    rm "$temp_env"
    
    # Define datasets and algorithms (same as main function)
    declare -A datasets
    datasets["gsm8k"]="$GSM8K_TEST_PATH"
    datasets["math500"]="$MATH500_PATH"
    datasets["aime"]="$AIME_PATH"
    
    declare -A algorithms
    algorithms["beamsearch"]="search/beamsearch/beamsearch.py"
    algorithms["beamsearch_merge"]="search/beamsearch/beamsearch_merge.py"
    algorithms["bfs"]="search/bfs/bfs.py"
    algorithms["bfs_merge"]="search/bfs/bfs_merge.py"
    algorithms["run_mcts"]="search/mcts/run_mcts.py"
    algorithms["run_mcts_merge"]="search/mcts/run_mcts_merge.py"
    algorithms["ssdp"]="search/SSDP/SSDP.py"
    
    total_experiments=$((${#datasets[@]} * ${#algorithms[@]}))
    
    echo "Would run the following experiments:"
    echo "Total experiments: $total_experiments"
    echo ""
    
    for dataset_name in "${!datasets[@]}"; do
        echo "Dataset: $dataset_name"
        echo "  Path: ${datasets[$dataset_name]}"
        for algorithm in "${!algorithms[@]}"; do
            echo "  - $algorithm (${algorithms[$algorithm]})"
        done
        echo ""
    done
    
    echo "Git commits would be made after each experiment with message:"
    echo "  'experiment ran for {algorithm}-{dataset}'"
}

# Main script logic
main() {
    # Parse command line arguments
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            dry_run
            exit 0
            ;;
        --check)
            check_prerequisites
            exit 0
            ;;
        "")
            # No arguments, run experiments
            check_prerequisites
            run_experiments
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
