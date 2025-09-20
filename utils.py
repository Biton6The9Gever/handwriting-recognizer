import os
import shutil

class Utils:
    @staticmethod
    def recreate_data_folder():
        folder_path=rf'..\ML_Project\Dataset\Processed'
        
        # Recreate the folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(rf'{folder_path}', exist_ok=True)
            print(f"Recreated folder: {folder_path}")
        


