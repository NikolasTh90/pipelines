import sys
import subprocess
import time

def install_package(package_name):
    """
    Install a Python package using pip
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        print(f"Installing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        time.sleep(2)  # Give a moment for installation to complete
        print(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {str(e)}")
        print(f"To install manually, run: pip install {package_name}")
        return False
    except Exception as e:
        print(f"Unexpected error during installation: {str(e)}")
        return False

def install_hub_validator(validator_name):
    """
    Install a validator from Guardrails hub
    
    Args:
        validator_name: Name of the validator to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        # Install system dependencies (in case they're needed)
        print(f"Installing system dependencies for {validator_name}...")
        try:
            subprocess.check_call(["apt-get", "update", "-y"])
            subprocess.check_call([
                "apt-get", "install", "-y",
                "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
                "libxext6", "libxrender-dev"
            ])
            print("System dependencies installed successfully")
        except Exception as e:
            print(f"Could not install system dependencies: {e}")
            print("Continuing with hub installation anyway...")
        
        # Install validator from hub
        print(f"Installing {validator_name} validator from Guardrails hub...")
        subprocess.check_call(["guardrails", "hub", "install", f"hub://guardrails/{validator_name}"])
        time.sleep(2)  # Wait for installation
        
        # Try direct pip installation as fallback
        try:
            print(f"Also trying direct pip installation of {validator_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", f"guardrails-hub-{validator_name.replace('_', '-')}"
            ])
        except Exception as e:
            print(f"Direct pip installation failed: {e}")
            print("Continuing with hub installation only")

        print(f"{validator_name} installation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{validator_name} installation failed: {str(e)}")
        print(f"To install manually, run: guardrails hub install hub://guardrails/{validator_name}")
        return False
    except Exception as e:
        print(f"Unexpected error during {validator_name} installation: {str(e)}")
        return False
