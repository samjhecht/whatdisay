#!/usr/bin/env python3

import sys
import os
import yaml


class Config:

    def __init__(self):
        self.config_path = os.path.join(os.path.abspath(os.path.dirname(__name__)), "config.yaml")
        self.loader = yaml.SafeLoader

    def configure(self):
        config = {}
        config['DEEPGRAM_API_KEY'] = input("Enter your Deepgram API Key (https://deepgram.com/). : ")
        config['HUGGINGFACE_TOKEN'] = input("Enter your Huggingface API Token (https://huggingface.co/). : ")
        config['WHISPER_MODEL'] = input(f"Whisper model type (defaults to 'large' if no value is passed). :") or "large"
        config['DEEPGRAM_MODEL'] = input(f"Deepgram model type (defaults to 'meeting' if no value is passed). :") or "meeting"
        
        output_default = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__name__))) + '/output/'
        config['OUTPUT_DIR'] = input(f"Optionally specify a directory for all task output to be saved. Defaults to creating 'output' directory in current working directory if no location specified.:") or output_default

        # Save to a yaml file in the project root directory.
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f'Config file saved at {self.config_path}')

    def get_config(self) -> dict:

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.load(f, self.loader)
        except FileNotFoundError:
            print("Library not configured.  Run the CLI with the '--configure' argument first.")
            sys.exit(1)
        
        return config


    def get_param(self,p) -> str:

        config = self.get_config()

        try:
            if p in config.keys() and config[p]:
                return config[p]
            else:
                raise ValueError(f"No config value set for {p}.  Run '--configure' to configure.")
        except AttributeError:
            print(f"No config value set for {p}.  Run '--configure' to configure.")
            sys.exit(1)
