import os

from logging_support import log_info


class DirectorySupport:

    @staticmethod
    def create_directory(abs_output_dir):
        # Create output directory if it doesn't exist
        if not os.path.exists(abs_output_dir):
            log_info(f"Creating output directory {abs_output_dir}")
            os.umask(0)
            os.makedirs(abs_output_dir)
