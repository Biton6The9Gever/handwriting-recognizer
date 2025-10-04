from Scripts.utils import Utils

Utils.create_dataset()
print('starting analysis of dataset...')
Utils.get_images_size_distribution(Utils.DATA_PATH)
print('done analysis of dataset.')
Utils.recreate_data_folder()
Utils.create_csv_file()