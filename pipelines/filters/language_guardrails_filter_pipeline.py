import os
import sys
import subprocess
import importlib.util
import time
import traceback
from typing import Optional, List
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Language_Filter")

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to
        pipelines: List[str] = ["*"]
        
        # Assign a priority level to the filter pipeline
        priority: int = 0
        
        # Custom configuration for language detection and translation
        expected_languages_iso: List[str] = ["en", "el"]  # Default expected languages: English and Greek
        threshold: float = 0.75  # Confidence threshold for language detection
        log_level: str = "info"
        enforce_language: bool = True  # If True, enforces translation when language differs
        adapt_to_user_language: bool = True  # If True, tries to translate to the user's detected language
        on_fail: str = "fix"  # 'fix' will translate, 'exception' will block, etc.
        use_gpu: bool = True  # Whether to use GPU for translation
        
        # Fallback options when Guardrails isn't available
        use_fallback_detection: bool = True
        
        # Installation options
        attempt_auto_install: bool = True
        
    def __init__(self):
        # This filter uses Guardrails AI's CorrectLanguage module to detect and translate content
        self.type = "filter"
        self.name = "Guardrails Language Filter"

        # Initialize with default configuration
        self.valves = self.Valves()
        
        # Set device for GPU/CPU
        if self.valves.use_gpu:
            os.environ["DEVICE"] = "cuda"
            logger.info("Set language validator to use GPU (CUDA)")
        else:
            os.environ["DEVICE"] = "cpu" 
            logger.info("Set language validator to use CPU")
        
        # Print diagnostic information
        self._print_environment_info()
        
        # Set logging level based on configuration
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
        
        # Try to import Guardrails components
        logger.info("Initializing Guardrails Language Filter...")
        self.guardrails_mode = self._setup_guardrails()
        
        # Setup enhanced fallback if guardrails not available or as additional layer
        if self.valves.use_fallback_detection:
            self._setup_enhanced_fallback()
            logger.info("Enhanced fallback detection set up successfully")

    def _print_environment_info(self):
        """Print diagnostic information about the environment"""
        logger.info("=== Environment Information ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"DEVICE: {os.environ.get('DEVICE', 'Not set')}")
        logger.info(f"sys.path: {sys.path}")
        
        # Check GPU availability if using GPU
        if self.valves.use_gpu:
            try:
                import torch
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            except ImportError:
                logger.warning("Torch not available - cannot check CUDA status")
        
        # Check if guardrails is in the path
        guardrails_locations = [p for p in sys.path if 'guardrails' in p]
        if guardrails_locations:
            logger.info(f"Potential guardrails locations in path: {guardrails_locations}")
        
        # List available modules
        try:
            import pkgutil
            guardrails_specs = [p for p in pkgutil.iter_modules() if 'guardrails' in p.name]
            logger.info(f"Available guardrails modules: {guardrails_specs}")
        except Exception as e:
            logger.warning(f"Error listing modules: {e}")
        
        logger.info("===============================")

    def _setup_guardrails(self):
        """Set up Guardrails and return the mode (guardrails, fallback, or disabled)"""
        setup_start_time = time.time()
        logger.info("Setting up Guardrails language validator...")
        
        # First check if guardrails is installed
        try:
            import guardrails
            logger.info(f"Guardrails package found. Checking version...")
            try:
                logger.info(f"Guardrails version: {guardrails.__version__}")
            except AttributeError:
                logger.warning("Could not determine Guardrails version")
        except ImportError as e:
            logger.warning(f"Guardrails package not found: {e}")
            # Attempt installation if configured
            if self.valves.attempt_auto_install:
                logger.info("Attempting to install Guardrails...")
                if self._install_guardrails():
                    try:
                        import guardrails
                        logger.info("Guardrails successfully installed and imported")
                    except ImportError as e2:
                        logger.error(f"Could not import Guardrails after installation: {e2}")
                        setup_end_time = time.time()
                        logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
                        return "fallback"
                else:
                    logger.error("Failed to install Guardrails")
                    setup_end_time = time.time()
                    logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
                    return "fallback"
            else:
                setup_end_time = time.time()
                logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
                return "fallback"
        
        # Now check for Guard class
        try:
            from guardrails import Guard
            logger.info("Guardrails Guard class successfully imported")
        except ImportError as e:
            logger.error(f"Could not import Guard from guardrails: {e}")
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
            return "fallback"
        
        # Now check for hub module and contents
        try:
            from guardrails import hub
            logger.info("Guardrails hub module found")
            logger.info(f"Hub contents: {dir(hub)}")
        except ImportError as e:
            logger.error(f"Could not import hub from guardrails: {e}")
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
            return "fallback"
            
        # Try to import CorrectLanguage specifically - try multiple possible locations
        correct_language_module_found = False
        for import_path in [
            "from guardrails.hub import CorrectLanguage",
            "from guardrails_hub_correct_language import CorrectLanguage",
            "from guardrails_grhub_correct_language import CorrectLanguage"
        ]:
            try:
                logger.info(f"Trying import: {import_path}")
                exec(import_path, globals())
                logger.info("CorrectLanguage successfully imported!")
                correct_language_module_found = True
                break
            except ImportError as e:
                logger.warning(f"Import failed: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during import: {e}")
        
        if not correct_language_module_found:
            logger.error("Could not import CorrectLanguage from any known location")
            # Last attempt - try to install from hub directly
            if self.valves.attempt_auto_install:
                logger.info("Attempting to install CorrectLanguage validator from hub...")
                if self._install_correct_language():
                    # Try imports again
                    for import_path in [
                        "from guardrails.hub import CorrectLanguage",
                        "from guardrails_hub_correct_language import CorrectLanguage",
                        "from guardrails_grhub_correct_language import CorrectLanguage"
                    ]:
                        try:
                            logger.info(f"Trying import after installation: {import_path}")
                            exec(import_path, globals())
                            logger.info("CorrectLanguage successfully imported after installation!")
                            correct_language_module_found = True
                            break
                        except ImportError as e:
                            logger.warning(f"Import failed after installation: {e}")
                        except Exception as e:
                            logger.warning(f"Unexpected error during import after installation: {e}")
            
            if not correct_language_module_found:
                logger.error("CorrectLanguage could not be imported after all attempts")
                setup_end_time = time.time()
                logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
                return "fallback"
        
        # Try to instantiate CorrectLanguage and create Guard
        try:
            # Test instantiation - store the on_fail mode to handle validation results appropriately
            logger.info(f"Attempting to instantiate CorrectLanguage with on_fail={self.valves.on_fail}...")
            
            # Store the original on_fail mode for later reference
            self.on_fail_mode = self.valves.on_fail
            logger.info(f"Using on_fail mode: {self.on_fail_mode}")
            
            # Use the first language in the list for the default validator
            validator = CorrectLanguage(
                expected_language_iso=self.valves.expected_languages_iso[0],
                threshold=self.valves.threshold,
                on_fail=self.on_fail_mode
            )
            logger.info(f"CorrectLanguage validator created: {validator}")
            
            # Create Guard
            from guardrails import Guard
            logger.info("Creating Guard with CorrectLanguage validator...")
            self.guard = Guard().use(
                CorrectLanguage,
                expected_language_iso=self.valves.expected_languages_iso[0],  # Use first language in list
                threshold=self.valves.threshold,
                on_fail=self.on_fail_mode
            )
            logger.info("Guard successfully created with CorrectLanguage")
            
            # Test validation with a simple string
            try:
                logger.info("Testing validator with a simple English string...")
                validation_start = time.time()
                result = self.guard.validate("This is a test string.")
                validation_end = time.time()
                
                # Log the validation result in a way that works for all on_fail modes
                logger.info(f"Validation outcome: validation_passed={result.validation_passed}")
                logger.info(f"Original text: 'This is a test string.'")
                logger.info(f"Validation test took {validation_end - validation_start:.2f} seconds")
                
                # Check if validated_output exists and is not None before logging
                if hasattr(result, 'validated_output') and result.validated_output is not None:
                    logger.info(f"Validated output: '{result.validated_output}'")
                else:
                    logger.info("Validated output: None")
                
                # Log if the device is actually being used
                logger.info(f"Device being used: {os.environ.get('DEVICE', 'Not reported')}")
            except Exception as e:
                logger.warning(f"Validation test failed: {e}")
                logger.warning("This may indicate issues with the validator, but we'll continue")
                # Don't return fallback yet, as validation might work for real inputs
            
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (guardrails mode)")
            return "guardrails"
        except Exception as e:
            logger.error(f"Error setting up CorrectLanguage or Guard: {e}")
            logger.error("Full traceback:")
            traceback.print_exc()
            setup_end_time = time.time()
            logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds (fallback mode)")
            return "fallback"
                
    def _install_guardrails(self):
        """Attempt to install Guardrails using subprocess"""
        install_start = time.time()
        try:
            # First install the guardrails-ai package
            logger.info("Installing guardrails-ai package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "guardrails-ai"])
            
            # Wait a moment for installation to complete
            time.sleep(2)
            install_end = time.time()
            logger.info(f"Guardrails-ai package installation completed in {install_end - install_start:.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Installation failed: {str(e)}")
            logger.info("To install manually, run:")
            logger.info("  pip install guardrails-ai")
            install_end = time.time()
            logger.info(f"Installation failed after {install_end - install_start:.2f} seconds")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {str(e)}")
            install_end = time.time()
            logger.info(f"Installation failed after {install_end - install_start:.2f} seconds")
            return False

    def _install_correct_language(self):
        """Attempt to install CorrectLanguage validator from hub"""
        install_start = time.time()
        try:
            # Install system dependencies (in case they're needed)
            logger.info("Installing system dependencies for CorrectLanguage...")
            try:
                subprocess.check_call([
                    "apt-get", "update", "-y"
                ])
                subprocess.check_call([
                    "apt-get", "install", "-y",
                    "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
                    "libxext6", "libxrender-dev"
                ])
                logger.info("System dependencies installed successfully")
            except Exception as e:
                logger.warning(f"Could not install system dependencies: {e}")
                logger.warning("Continuing with hub installation anyway...")
            
            # Install required Python dependencies
            logger.info("Installing required Python dependencies...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "fast-langdetect", "iso-language-codes", "transformers"
                ])
                logger.info("Python dependencies installed successfully")
            except Exception as e:
                logger.warning(f"Could not install Python dependencies: {e}")
                logger.warning("This might cause issues with the validator")
            
            # Install CorrectLanguage from hub
            logger.info("Installing CorrectLanguage validator from Guardrails hub...")
            subprocess.check_call(["guardrails", "hub", "install", "hub://scb-10x/correct_language"])
            time.sleep(2)

            # Try direct pip installation as fallback
            try:
                logger.info("Also trying direct pip installation of correct_language...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "guardrails-hub-correct-language"
                ])
            except Exception as e:
                logger.warning(f"Direct pip installation failed: {e}")
                logger.warning("Continuing with hub installation only")

            install_end = time.time()
            logger.info(f"CorrectLanguage installation completed in {install_end - install_start:.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"CorrectLanguage installation failed: {str(e)}")
            logger.error("To install manually, run:")
            logger.error("  guardrails hub install hub://scb-10x/correct_language")
            install_end = time.time()
            logger.info(f"Installation failed after {install_end - install_start:.2f} seconds")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during CorrectLanguage installation: {str(e)}")
            install_end = time.time()
            logger.info(f"Installation failed after {install_end - install_start:.2f} seconds")
            return False

    def _setup_enhanced_fallback(self):
        """Set up enhanced fallback language detection and translation"""
        setup_start = time.time()
        logger.info("Setting up enhanced fallback language detection and translation")
        
        # Try to set up a fallback language detector and translator
        try:
            # Try to import fast_langdetect for language detection
            try:
                import fast_langdetect
                self.has_langdetect = True
                self.detect_language = fast_langdetect.detect
                logger.info("Successfully set up fast_langdetect for fallback detection")
            except ImportError:
                self.has_langdetect = False
                logger.warning("fast_langdetect not available, trying to install...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "fast-langdetect"
                    ])
                    import fast_langdetect
                    self.has_langdetect = True
                    self.detect_language = fast_langdetect.detect
                    logger.info("Successfully installed and set up fast_langdetect")
                except Exception as e:
                    logger.warning(f"Could not install fast_langdetect: {e}")
                    logger.warning("Fallback detection will be limited")
            
            # Try to import or install iso-language-codes
            try:
                import iso_language_codes
                self.has_iso_codes = True
                logger.info("Successfully set up iso_language_codes for fallback language names")
            except ImportError:
                self.has_iso_codes = False
                logger.warning("iso_language_codes not available, trying to install...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "iso-language-codes"
                    ])
                    import iso_language_codes
                    self.has_iso_codes = True
                    logger.info("Successfully installed and set up iso_language_codes")
                except Exception as e:
                    logger.warning(f"Could not install iso_language_codes: {e}")
            
            # Try to set up a simple translation service using transformers if available
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                
                logger.info("Setting up translation model using transformers...")
                # Lazy loading - we'll load the model only when needed
                self.has_transformers = True
                self.translation_model_setup = False
                
                def setup_translation_model():
                    """Set up the translation model on demand"""
                    model_setup_start = time.time()
                    logger.info("Loading translation model (first use)...")
                    self.translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
                    self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
                    self.translation_model_setup = True
                    model_setup_end = time.time()
                    logger.info(f"Translation model loaded successfully in {model_setup_end - model_setup_start:.2f} seconds")
                
                def translate_text(text, src_lang, tgt_lang="eng_Latn"):
                    """Translate text using the NLLB model"""
                    translate_start = time.time()
                    if not self.translation_model_setup:
                        setup_translation_model()
                    
                    # Convert language code to NLLB format if needed
                    src_lang_nllb = self.nllb_lang_map.get(src_lang, src_lang)
                    tgt_lang_nllb = self.nllb_lang_map.get(tgt_lang, "eng_Latn")  # Default to English if not found
                    
                    logger.info(f"Translating from {src_lang_nllb} to {tgt_lang_nllb}")
                    
                    # Tokenize and translate
                    inputs = self.translation_tokenizer(text, return_tensors="pt")
                    translation_tokens = self.translation_model.generate(
                        **inputs,
                        forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang_nllb],
                        max_length=200
                    )
                    translation = self.translation_tokenizer.batch_decode(
                        translation_tokens, skip_special_tokens=True
                    )[0]
                    
                    translate_end = time.time()
                    logger.info(f"Translation completed in {translate_end - translate_start:.2f} seconds")
                    return translation
                
                self.translate_text = translate_text
                logger.info("Translation function set up successfully with transformers")
            except ImportError:
                self.has_transformers = False
                logger.warning("transformers not available for translation, trying to install...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "transformers"
                    ])
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                    self.has_transformers = True
                    self.translation_model_setup = False
                    
                    def setup_translation_model():
                        """Set up the translation model on demand"""
                        model_setup_start = time.time()
                        logger.info("Loading translation model (first use)...")
                        self.translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
                        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
                        self.translation_model_setup = True
                        model_setup_end = time.time()
                        logger.info(f"Translation model loaded successfully in {model_setup_end - model_setup_start:.2f} seconds")
                    
                    def translate_text(text, src_lang, tgt_lang="eng_Latn"):
                        """Translate text using the NLLB model"""
                        translate_start = time.time()
                        if not self.translation_model_setup:
                            setup_translation_model()
                        
                        # Convert language code to NLLB format if needed
                        src_lang_nllb = self.nllb_lang_map.get(src_lang, src_lang)
                        tgt_lang_nllb = self.nllb_lang_map.get(tgt_lang, "eng_Latn")  # Default to English if not found
                        
                        logger.info(f"Translating from {src_lang_nllb} to {tgt_lang_nllb}")
                        
                        # Tokenize and translate
                        inputs = self.translation_tokenizer(text, return_tensors="pt")
                        translation_tokens = self.translation_model.generate(
                            **inputs,
                            forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang_nllb],
                            max_length=200
                        )
                        translation = self.translation_tokenizer.batch_decode(
                            translation_tokens, skip_special_tokens=True
                        )[0]
                        
                        translate_end = time.time()
                        logger.info(f"Translation completed in {translate_end - translate_start:.2f} seconds")
                        return translation
                    
                    self.translate_text = translate_text
                    logger.info("Successfully installed and set up transformers for translation")
                except Exception as e:
                    logger.warning(f"Could not install transformers: {e}")
                    logger.warning("Fallback translation will not be available")
                    self.has_transformers = False
            
            # Set up language mapping for NLLB
            self.nllb_lang_map = {
                "en": "eng_Latn",  # English
                "fr": "fra_Latn",  # French
                "es": "spa_Latn",  # Spanish
                "de": "deu_Latn",  # German
                "it": "ita_Latn",  # Italian
                "pt": "por_Latn",  # Portuguese
                "nl": "nld_Latn",  # Dutch
                "ru": "rus_Cyrl",  # Russian
                "zh": "zho_Hans",  # Chinese (Simplified)
                "ja": "jpn_Jpan",  # Japanese
                "ko": "kor_Hang",  # Korean
                "ar": "ara_Arab",  # Arabic
                "hi": "hin_Deva",  # Hindi
                "bn": "ben_Beng",  # Bengali
                "ur": "urd_Arab",  # Urdu
                "th": "tha_Thai",  # Thai
                "vi": "vie_Latn",  # Vietnamese
                "sw": "swh_Latn",  # Swahili
                "tr": "tur_Latn",  # Turkish
                "pl": "pol_Latn",  # Polish
                "uk": "ukr_Cyrl",  # Ukrainian
                "cs": "ces_Latn",  # Czech
                "sk": "slk_Latn",  # Slovak
                "hu": "hun_Latn",  # Hungarian
                "ro": "ron_Latn",  # Romanian
                "fi": "fin_Latn",  # Finnish
                "da": "dan_Latn",  # Danish
                "sv": "swe_Latn",  # Swedish
                "no": "nno_Latn",  # Norwegian
                "el": "ell_Grek",  # Greek
                "bg": "bul_Cyrl",  # Bulgarian
                "mk": "mkd_Cyrl",  # Macedonian
                "sr": "srp_Cyrl",  # Serbian
                "be": "bel_Cyrl",  # Belarusian
            }
            logger.info(f"NLLB language mapping set up with {len(self.nllb_lang_map)} languages")
            
            # Determine if we have a complete fallback system
            self.has_complete_fallback = self.has_langdetect and self.has_transformers
            if self.has_complete_fallback:
                logger.info("Complete fallback language detection and translation setup successful")
            else:
                logger.warning("Incomplete fallback system - some features may not work")
                logger.warning(f"Language detection: {'Available' if self.has_langdetect else 'Missing'}")
                logger.warning(f"Translation capability: {'Available' if self.has_transformers else 'Missing'}")
                
        except Exception as e:
            logger.error(f"Error setting up fallback language system: {e}")
            logger.error("Fallback language detection and translation will not be available")
            self.has_langdetect = False
            self.has_transformers = False
            self.has_complete_fallback = False
        
        setup_end = time.time()
        logger.info(f"Enhanced fallback setup completed in {setup_end - setup_start:.2f} seconds")

    async def on_startup(self):
        # This function is called when the server is started
        logger.info(f"Guardrails Language Filter starting up")
        
        if self.guardrails_mode == "guardrails":
            logger.info(f"Running with Guardrails CorrectLanguage validator")
            logger.info(f"Expected languages: {self.valves.expected_languages_iso}")
            logger.info(f"Detection threshold: {self.valves.threshold}")
            logger.info(f"On fail action: {self.valves.on_fail}")
            logger.info(f"Using {'GPU' if self.valves.use_gpu else 'CPU'} for detection and translation")
            logger.info(f"Adapt to user language: {self.valves.adapt_to_user_language}")
        elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection:
            logger.info("Running in fallback mode with enhanced language detection and translation")
            if hasattr(self, 'has_langdetect') and self.has_langdetect:
                logger.info("Using fast_langdetect for language detection")
            if hasattr(self, 'has_transformers') and self.has_transformers:
                logger.info("Using transformers for translation")
            logger.info("For better detection and translation, install the full Guardrails CorrectLanguage validator:")
            logger.info("  pip install guardrails-ai")
            logger.info("  guardrails hub install hub://scb-10x/correct_language")
        else:
            logger.info("Language detection and translation disabled")
            
        logger.info(f"Enforce language: {self.valves.enforce_language}")

    async def on_shutdown(self):
        # This function is called when the server is stopped
        logger.info("Guardrails Language Filter shutting down")

    async def on_valves_updated(self):
        # This function is called when the valves are updated
        logger.info(f"Valves updated: Expected languages = {self.valves.expected_languages_iso}")
        logger.info(f"Detection threshold: {self.valves.threshold}")
        logger.info(f"Enforce language: {self.valves.enforce_language}")
        logger.info(f"Adapt to user language: {self.valves.adapt_to_user_language}")
        logger.info(f"On fail action: {self.valves.on_fail}")
        logger.info(f"Using GPU: {self.valves.use_gpu}")
        
        # If on_fail mode changed, store it
        if hasattr(self, 'on_fail_mode') and self.on_fail_mode != self.valves.on_fail:
            self.on_fail_mode = self.valves.on_fail
            logger.info(f"Updated on_fail mode to: {self.on_fail_mode}")
        
        # Update GPU/CPU setting if it changed
        if self.valves.use_gpu:
            os.environ["DEVICE"] = "cuda"
            logger.info("Set language validator to use GPU (CUDA)")
        else:
            os.environ["DEVICE"] = "cpu" 
            logger.info("Set language validator to use CPU")
        
        # Update the logging level if it changed
        if hasattr(logging, self.valves.log_level.upper()):
            logger.setLevel(getattr(logging, self.valves.log_level.upper()))
            logger.info(f"Updated log level to {self.valves.log_level.upper()}")
        
        # Update the validator if we're using Guardrails
        if self.guardrails_mode == "guardrails":
            try:
                # We use the global CorrectLanguage that was successfully imported earlier
                from guardrails import Guard
                
                # Recreate the Guard with updated parameters - use first expected language
                self.guard = Guard().use(
                    CorrectLanguage,
                    expected_language_iso=self.valves.expected_languages_iso[0],  # Use first language
                    threshold=self.valves.threshold,
                    on_fail=self.valves.on_fail
                )
                logger.info(f"Updated Guardrails Guard with new parameters")
            except Exception as e:
                logger.error(f"Error updating Guardrails Guard: {str(e)}")
                traceback.print_exc()
                logger.warning("Falling back to enhanced language detection")
                self.guardrails_mode = "fallback"
                self._setup_enhanced_fallback()
        
        # Always ensure fallback is set up, whether we're in fallback mode or not
        if self.valves.use_fallback_detection:
            self._setup_enhanced_fallback()

    def _detect_language_fallback(self, text):
        """Fallback language detection using fast_langdetect if available"""
        detect_start = time.time()
        if not hasattr(self, 'has_langdetect') or not self.has_langdetect:
            logger.warning("No fallback language detection available")
            detect_end = time.time()
            logger.debug(f"Detection attempt failed in {detect_end - detect_start:.2f} seconds")
            return None, 0.0  # Return None language and 0.0 confidence
        
        try:
            # Detect the language
            detected = self.detect_language(text)
            if detected:
                lang_code = detected
                confidence = 1.0  # fast_langdetect doesn't provide confidence, so we assume high confidence
                
                # Get language name if iso_language_codes is available
                language_name = "unknown"
                if hasattr(self, 'has_iso_codes') and self.has_iso_codes:
                    try:
                        import iso_language_codes
                        language_info = iso_language_codes.get_language_info(lang_code)
                        if language_info:
                            language_name = language_info.get('name', 'unknown')
                    except Exception as e:
                        logger.warning(f"Error getting language name: {e}")
                
                detect_end = time.time()
                logger.info(f"Detected language: {lang_code} ({language_name}) with confidence: {confidence:.2f} in {detect_end - detect_start:.2f} seconds")
                return lang_code, confidence
            else:
                logger.warning("Language detection failed")
                detect_end = time.time()
                logger.debug(f"Detection failed in {detect_end - detect_start:.2f} seconds")
                return None, 0.0
        except Exception as e:
            logger.error(f"Error in fallback language detection: {e}")
            detect_end = time.time()
            logger.debug(f"Detection error in {detect_end - detect_start:.2f} seconds")
            return None, 0.0

    def _translate_text_fallback(self, text, detected_lang, target_lang):
        """Fallback translation using transformers if available"""
        translate_start = time.time()
        if not hasattr(self, 'has_transformers') or not self.has_transformers:
            logger.warning("No fallback translation available")
            translate_end = time.time()
            logger.debug(f"Translation attempt failed in {translate_end - translate_start:.2f} seconds")
            return text  # Return original text if translation is not available
        
        try:
            # Convert to NLLB format using the mapping
            src_lang_nllb = self.nllb_lang_map.get(detected_lang, detected_lang)
            tgt_lang_nllb = self.nllb_lang_map.get(target_lang, "eng_Latn")  # Default to English if not found
            
            logger.info(f"Translating from {detected_lang} ({src_lang_nllb}) to {target_lang} ({tgt_lang_nllb})")
            
            # Perform translation
            translated_text = self.translate_text(text, src_lang_nllb, tgt_lang_nllb)
            
            if translated_text:
                translate_end = time.time()
                logger.info(f"Translation successful in {translate_end - translate_start:.2f} seconds")
                logger.debug(f"Original text: '{text[:50]}...'")
                logger.debug(f"Translated text: '{translated_text[:50]}...'")
                return translated_text
            else:
                logger.warning("Translation returned empty result")
                translate_end = time.time()
                logger.debug(f"Translation failed in {translate_end - translate_start:.2f} seconds")
                return text  # Return original text if translation failed
        except Exception as e:
            logger.error(f"Error in fallback translation: {e}")
            logger.error(traceback.format_exc())
            translate_end = time.time()
            logger.debug(f"Translation error in {translate_end - translate_start:.2f} seconds")
            return text  # Return original text if there was an error

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process incoming messages - detect user language in the inlet
        
        Args:
            body: The request body containing the messages
            user: Optional user information
            
        Returns:
            The original body with user language information
        """
        inlet_start = time.time()
        if "messages" not in body or not body["messages"]:
            logger.debug("No messages found in request body")
            inlet_end = time.time()
            logger.debug(f"Inlet processing completed in {inlet_end - inlet_start:.2f} seconds")
            return body
            
        username = user.get("username", "unknown") if user else "unknown"
        
        # Get the user's message
        user_message = None
        for message in reversed(body["messages"]):
            if message.get("role") == "user":
                user_message = message.get("content")
                break
        
        if user_message:
            logger.info(f"Processing message from {username}: {user_message[:30]}...")
            
            # Detect the language of the user's message
            detect_start = time.time()
            
            if hasattr(self, 'has_langdetect') and self.has_langdetect:
                detected_lang, confidence = self._detect_language_fallback(user_message)
                if detected_lang and confidence >= self.valves.threshold:
                    # Store the detected language in the body for later use
                    if "metadata" not in body:
                        body["metadata"] = {}
                    body["metadata"]["user_language"] = detected_lang
                    logger.info(f"User language detected: {detected_lang} with confidence {confidence:.2f}")
            
            detect_end = time.time()
            logger.debug(f"User language detection completed in {detect_end - detect_start:.2f} seconds")
        
        # Pass the (possibly modified) body through
        inlet_end = time.time()
        logger.debug(f"Inlet processing completed in {inlet_end - inlet_start:.2f} seconds")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process outgoing messages - check if the language is correct and translate if needed
        
        Args:
            body: The response body containing the model's output
            user: Optional user information
            
        Returns:
            The body with potentially translated content
        """
        outlet_start = time.time()
        if "choices" not in body or not body["choices"]:
            logger.debug("No choices found in response body")
            outlet_end = time.time()
            logger.debug(f"Outlet processing completed in {outlet_end - outlet_start:.2f} seconds (no choices)")
            return body
        
        # Get user's detected language if available
        user_language = None
        if "metadata" in body and "user_language" in body["metadata"]:
            user_language = body["metadata"]["user_language"]
            logger.info(f"Found user language from metadata: {user_language}")
        
        # Check each choice for its language
        for i, choice in enumerate(body["choices"]):
            # Skip empty messages
            if "message" not in choice or "content" not in choice["message"] or not choice["message"]["content"]:
                continue
                
            message_content = choice["message"]["content"]
            username = user.get("username", "unknown") if user else "unknown"
            logger.info(f"Processing response for {username}: {message_content[:30]}...")
            
            # Check language and translate if needed
            is_wrong_language = False
            error_message = ""
            detected_lang = None
            
            # Detect the language of the response
            detect_start = time.time()
            if hasattr(self, 'has_langdetect') and self.has_langdetect:
                detected_lang, confidence = self._detect_language_fallback(message_content)
                detect_end = time.time()
                logger.info(f"Response language detected: {detected_lang} with confidence {confidence:.2f} in {detect_end - detect_start:.2f} seconds")
            else:
                logger.warning("No language detection available for response")
                detected_lang = None
                detect_end = time.time()
                logger.debug(f"Detection attempt failed in {detect_end - detect_start:.2f} seconds")
            
            # If language detection failed, skip this choice
            if not detected_lang:
                logger.warning("Skipping translation due to failed language detection")
                continue
            
            # Use the appropriate translation method based on mode
            if self.guardrails_mode == "guardrails":
                logger.debug("Using Guardrails for language validation and translation")
                
                # Check if response language is in the list of expected languages
                if detected_lang not in self.valves.expected_languages_iso and confidence >= self.valves.threshold:
                    is_wrong_language = True
                    logger.info(f"Response language {detected_lang} not in expected languages {self.valves.expected_languages_iso}")
                    
                    # Determine target language for translation
                    target_lang = None
                    if self.valves.adapt_to_user_language and user_language:
                        # Use user's language if available and adaptation is enabled
                        target_lang = user_language
                        logger.info(f"Adapting to user language: {target_lang}")
                    else:
                        # Otherwise use the first expected language
                        target_lang = self.valves.expected_languages_iso[0]
                        logger.info(f"Using default expected language: {target_lang}")
                    
                    # Validate with Guardrails (will translate if needed)
                    translation_start = time.time()
                    try:
                        from guardrails import Guard
                        
                        # Create a temporary guard with the target language
                        temp_guard = Guard().use(
                            CorrectLanguage,
                            expected_language_iso=target_lang,
                            threshold=self.valves.threshold,
                            on_fail=self.valves.on_fail
                        )
                        
                        validation_result = temp_guard.validate(message_content)
                        
                        # Check if validation passed
                        if not validation_result.validation_passed:
                            logger.info(f"Translation needed from {detected_lang} to {target_lang}")
                            
                            # Get the translated text if available
                            if hasattr(validation_result, 'validated_output') and validation_result.validated_output:
                                translated_content = validation_result.validated_output
                                logger.info(f"Translated text: '{translated_content[:50]}...'")
                                
                                # Update the message with the translated content
                                if self.valves.enforce_language:
                                    logger.info(f"Replacing content with translated version")
                                    choice["message"]["content"] = translated_content
                            else:
                                logger.warning("No translated content available from validator")
                    except Exception as e:
                        # If translation to user language fails, try English
                        logger.warning(f"Error translating to {target_lang}: {e}")
                        if target_lang != "en":
                            logger.info("Attempting to translate to English as fallback")
                            try:
                                from guardrails import Guard
                                
                                fallback_guard = Guard().use(
                                    CorrectLanguage,
                                    expected_language_iso="en",
                                    threshold=self.valves.threshold,
                                    on_fail=self.valves.on_fail
                                )
                                
                                fallback_result = fallback_guard.validate(message_content)
                                
                                if not fallback_result.validation_passed and hasattr(fallback_result, 'validated_output') and fallback_result.validated_output:
                                    translated_content = fallback_result.validated_output
                                    logger.info(f"Fallback translation to English: '{translated_content[:50]}...'")
                                    
                                    if self.valves.enforce_language:
                                        choice["message"]["content"] = translated_content
                            except Exception as e2:
                                logger.error(f"Fallback translation also failed: {e2}")
                                # Try fallback translation method if available
                                if hasattr(self, 'has_transformers') and self.has_transformers:
                                    logger.info("Attempting translation using fallback method")
                                    translated_content = self._translate_text_fallback(
                                        message_content,
                                        detected_lang,
                                        "en"  # Fallback to English
                                    )
                                    if self.valves.enforce_language:
                                        choice["message"]["content"] = translated_content
                    
                    translation_end = time.time()
                    logger.info(f"Translation processing completed in {translation_end - translation_start:.2f} seconds")
                elif detected_lang:
                    logger.info(f"Response language {detected_lang} is in expected languages - no translation needed")
            
            elif self.guardrails_mode == "fallback" and self.valves.use_fallback_detection and hasattr(self, 'has_complete_fallback') and self.has_complete_fallback:
                # Use fallback detection and translation
                logger.debug("Using fallback language detection and translation")
                
                # Check if response language is in the list of expected languages
                if detected_lang not in self.valves.expected_languages_iso and confidence >= self.valves.threshold:
                    is_wrong_language = True
                    error_message = f"Content is in {detected_lang} instead of one of {self.valves.expected_languages_iso}"
                    logger.info(error_message)
                    
                    # Determine target language for translation
                    target_lang = None
                    if self.valves.adapt_to_user_language and user_language:
                        # Use user's language if available and adaptation is enabled
                        target_lang = user_language
                        logger.info(f"Adapting to user language: {target_lang}")
                    else:
                        # Otherwise use the first expected language
                        target_lang = self.valves.expected_languages_iso[0]
                        logger.info(f"Using default expected language: {target_lang}")
                    
                    # Translate the content if we're enforcing language
                    translation_start = time.time()
                    if self.valves.enforce_language:
                        translated_content = self._translate_text_fallback(
                            message_content, 
                            detected_lang, 
                            target_lang
                        )
                        
                        if translated_content != message_content:
                            logger.info(f"Translated text: '{translated_content[:50]}...'")
                            choice["message"]["content"] = translated_content
                        else:
                            logger.warning(f"Translation to {target_lang} failed, trying English")
                            if target_lang != "en":
                                # Try English as fallback
                                translated_content = self._translate_text_fallback(
                                    message_content, 
                                    detected_lang, 
                                    "en"
                                )
                                if translated_content != message_content:
                                    logger.info(f"Fallback translation to English: '{translated_content[:50]}...'")
                                    choice["message"]["content"] = translated_content
                    translation_end = time.time()
                    logger.info(f"Translation processing completed in {translation_end - translation_start:.2f} seconds")
                elif detected_lang:
                    logger.info(f"Response language {detected_lang} is in expected languages - no translation needed")
            
            else:
                # No detection or translation available
                logger.info(f"Response processed (language detection disabled)")
        
        # Track and log total processing time
        outlet_end = time.time()
        logger.info(f"Total outlet processing completed in {outlet_end - outlet_start:.2f} seconds")
        
        # Return the potentially modified body
        return body