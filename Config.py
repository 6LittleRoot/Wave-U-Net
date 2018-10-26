import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    # Base configuration
    model_config = {# SET MUSDB PATH HERE, AND SET CCMIXTER PATH IN CCMixter.xml
                    "musdb_path" : "./musdb18_augmented",
                    # musdb setup_file, set to None to use default
                    "musdb_setup" : "mus_augmented.yaml",
                    # regular expressions hat are used to filter training, test and validation data
                    # training data will always be filtered to remove files that match the first match from the
                    # validation filter pattern
                    "filter_data" : {"train" : None, "test": None, "valid" : "(.*)_orig/.*"},

                    # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!                        
                    "estimates_path" : "./Source_Estimates", 

                    # set to true for using CCMixter dataset - see README.md for where to get it from
                    "use_CCMixter": False,
                    "model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 64, # Batch size
                    "max_epochs" : 200000, # Batch size
                    "gpu_device" : -1, # None -> CPU, -1 -> lock any free gpu
                    "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    'cache_size' : 16, # Number of audio excerpts that are cached to build batches from
                    'num_workers' : 5, # Number of processes reading audio and filling up the cache
                    "duration" : 2, # Duration in seconds of the audio excerpts in the cache. Has to be at least the output length of the network!
                    'min_replacement_rate' : 16,  # roughly: how many cache entries to replace at least per batch on average. Can be fractional
                    'num_layers' : 12, # How many U-Net layers
                    # For Wave-U-Net: Filter size of conv in downsampling block
                    #'filter_size' : 15,
                    # reduced for 8000kHz srate
                    'filter_size' : 5,
                    # For Wave-U-Net: Filter size of conv in upsampling block
                    'merge_filter_size' : 5,
                    # Number of filters for convolution in first layer of network
                    'num_initial_filters' : 24, 
                    # DESIRED number of time frames in the output waveform per samples (could be changed
                    # when using valid padding)
                    #"num_frames": 16384,
                    # reduced due to reduced sample rate
                    "num_frames": 8192, 
                    'expected_sr': 8192,  # Downsample all audio input to this sampling rate
                    'mono_downmix': True,  # Whether to downsample the audio input
                    # Type of output layer, either "direct" or "difference". Direct output: Each source is result of tanh activation and independent. DIfference: Last source output is equal to mixture input - sum(all other sources)
                    'output_type' : 'direct',
                    # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
                    'context' : False, 
                    # Type of network architecture, either unet (our model) or unet_spectrogram (Jansson et al 2017 model)
                    'network' : 'unet', 
                    # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'upsampling' : 'learned', 
                    # Type of separation task. 'voice' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                    'task' : 'voice', 
                    'augmentation' : True, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                    # Only active for unet_spectrogram network. True: L2 loss on audio. False: L1 loss on spectrogram magnitudes for training and validation and test loss
                    'raw_audio_loss' : True,
                    # Patience for early stoppping on validation set
                    'worse_epochs' : 20, 
                    }
    seed=1337
    experiment_id = np.random.randint(0,1000000)
    # can be set on the command line to run evaluation of a checkpoint only
    eval_model = None
    model_config["num_sources"] = 4 if model_config["task"] == "multi_instrument" else 2
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2

@config_ingredient.named_config
def baseline():
    print("Training baseline model")

@config_ingredient.named_config
def baseline_diff():
    print("Training baseline model with difference output")
    model_config = {
        "output_type" : "difference"
    }

@config_ingredient.named_config
def baseline_context():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True
    }

@config_ingredient.named_config
def test_augmented():
    print("Training test model on augmented mono database with difference output and no input context (valid convolutions)")
    model_config = {
        "musdb_path" : "./musdb_test",
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False,
        "num_channels" : 1
    }

@config_ingredient.named_config
def baseline_stereo():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "mono_downmix" : False
    }

@config_ingredient.named_config
def full_mono_augmented():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and mono input/output, and learned upsampling layer")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False,
        "num_channels" : 1
    }

@config_ingredient.named_config
def full_mono_orig():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and mono input/output, and learned upsampling layer- use only original musdb18 files")
    model_config = {
        "filter_data" : {"train" : ".*_orig/.*", "test": None, "valid" : ".*_orig/.*"},
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False,
        "num_channels" : 1
    }
        
@config_ingredient.named_config
def full():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and stereo input/output, and learned upsampling layer")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False
    }

@config_ingredient.named_config
def baseline_context_smallfilter_deep():
    model_config = {
        "output_type": "difference",
        "context": True,
        "num_layers" : 14,
        "duration" : 7,
        "filter_size" : 5,
        "merge_filter_size" : 1
    }

@config_ingredient.named_config
def full_multi_instrument():
    print("Training multi-instrument separation with best model")
    model_config = {
        "output_type": "difference",
        "context": True,
        "upsampling": "linear",
        "mono_downmix": False,
        "task" : "multi_instrument"
    }

@config_ingredient.named_config
def baseline_comparison():
    model_config = {
        "batch_size": 4, # Less output since model is so big. Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "output_type": "difference",
        "context": True,
        "num_frames" : 768*127 + 1024,
        "duration" : 13,
        "expected_sr" : 8192,
        "num_initial_filters" : 34
    }

@config_ingredient.named_config
def unet_spectrogram():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16
    }

@config_ingredient.named_config
def unet_spectrogram_l1():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16,
        "loss" : "magnitudes"
    }
