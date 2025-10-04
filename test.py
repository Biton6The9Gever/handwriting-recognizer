from Scripts.utils import Utils

Utils.create_dataset()
Utils.get_images_size_distribution(Utils.DATA_PATH)
Utils.recreate_data_folder()
#Utils.create_csv_file()