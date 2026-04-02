#!/usr/bin/env python3
"""
Setup and configuration helper for the Streamlit documentation app
Checks dependencies, creates directories, and validates setup
"""

import sys
import subprocess
from pathlib import Path
import importlib


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ {text}")


def check_python_version():
    """Check Python version compatibility"""
    print_header("Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} (Compatible)")
        return True
    else:
        print_error(f"Python {version_str} (Requires Python 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Dependency Check")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'scipy': 'SciPy',
    }
    
    all_ok = True
    
    for package, display_name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{display_name} ({version})")
        except ImportError:
            print_error(f"{display_name} (NOT INSTALLED)")
            all_ok = False
    
    return all_ok


def create_directories():
    """Create necessary directories"""
    print_header("Directory Setup")
    
    project_root = Path(__file__).parent.parent.parent
    
    directories = [
        project_root / "streamlit" / "outputs",
        project_root / "checkpoints",
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory.relative_to(project_root)}/")
        else:
            print_info(f"Exists: {directory.relative_to(project_root)}/")


def check_model_files():
    """Check if model checkpoint exists"""
    print_header("Model Files Check")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir = project_root / "checkpoints"
    
    if not checkpoint_dir.exists():
        print_warning("Checkpoints directory doesn't exist yet")
        print_info("You need to train the model first:")
        print_info("  python -m training.train")
        return False
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if checkpoints:
        print_success(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            print_info(f"  - {ckpt.name}")
        return True
    else:
        print_warning("No checkpoints found in 'checkpoints/' directory")
        print_info("You need to train the model first:")
        print_info("  python -m training.train")
        return False


def check_data_files():
    """Check if data files exist"""
    print_header("Data Files Check")
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "dataset" / "data" / "x2"
    
    if not data_dir.exists():
        print_error(f"Data directory not found: {data_dir}")
        print_info("Please ensure dataset is properly downloaded")
        return False
    
    # Check for patches
    patch_dir = data_dir / "train_hr_patch"
    if patch_dir.exists():
        patches = list(patch_dir.glob("*.npy"))
        print_success(f"Found {len(patches)} training patches")
    else:
        print_warning("No training patches found")
    
    # Check for dataloaders
    dataloader_dir = data_dir / "dataload_filename"
    if dataloader_dir.exists():
        dataloaders = list(dataloader_dir.glob("*.txt"))
        print_success(f"Found {len(dataloaders)} dataloader files")
    else:
        print_warning("No dataloader files found")
    
    return True


def validate_config():
    """Validate configuration file"""
    print_header("Configuration Validation")
    
    try:
        from streamlit.config import AppConfig
        
        config = AppConfig()
        config_dict = {
            'img_size': config.img_size,
            'embed_dim': config.embed_dim,
            'encoder_depth': config.encoder_depth,
            'decoder_depth': config.decoder_depth,
            'default_device': config.default_device,
        }
        
        print_success("Configuration loaded successfully")
        print_info("Key settings:")
        print_info(f"  image_size: {config_dict['img_size']}×{config_dict['img_size']}")
        print_info(f"  embedding_dim: {config_dict['embed_dim']}")
        print_info(f"  encoder_depth: {config_dict['encoder_depth']}")
        print_info(f"  decoder_depth: {config_dict['decoder_depth']}")
        print_info(f"  default_device: {config_dict['default_device']}")
        
        return True
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return False


def generate_quickstart():
    """Generate quick start commands"""
    print_header("Quick Start Commands")
    
    print_info("To run the Streamlit app:")
    print("  streamlit run streamlit/app.py")
    print()
    print_info("To run inference examples:")
    print("  python streamlit/scripts/generate_examples.py")
    print()
    print_info("To train the model:")
    print("  python -m training.train")
    print()


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("  Streamlit Documentation App - Setup Checker")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Configuration", validate_config),
        ("Data Files", check_data_files),
        ("Model Files", check_model_files),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("Setup Summary")
    
    for name, result in results.items():
        status = "✓ OK" if result else "⚠ Needs Attention"
        print(f"  {name:20s}: {status}")
    
    critical_checks = ["Python Version", "Dependencies", "Configuration"]
    critical_ok = all(results.get(check, False) for check in critical_checks)
    
    if critical_ok:
        print_success("All critical checks passed!")
    else:
        print_error("Some critical checks failed. Please fix them before running the app.")
    
    generate_quickstart()
    
    print("=" * 60)
    print()
    
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
