import glob
import json
import os


def read_file_content(file_path):
    """reads the given json file content
        Args:
            file_path: The path of the json file.
        Returns:
            file_path: A list of stories read from the json file.
    """
    list_of_story_bodies = []
    # Open the json file.
    story_file = open(file_path)
    # convert the json file content into a dictionary.
    file_content = json.load(story_file)

    # Iterating through the json
    # list
    for story in file_content:
        list_of_story_bodies.append(story["body"])
    # Closing file
    story_file.close()

    return list_of_story_bodies


def get_list_all_stories():
    """get a list of all files in the following "path" and creates a merged list of stories.
        Args:
        Returns:
            all_stories: A merged list of stories.
    """
    path = '/Users/heshankavinda/Library/CloudStorage/OneDrive-UniversityofPlymouth/PROJ518/Project/First_Set_of_Algo/newDataset'
    all_stories = []
    for filename in glob.glob(os.path.join(path, '*.json')):
        all_stories.extend(read_file_content(filename))

    print(len(glob.glob(os.path.join(path, '*.json'))), "files opened and ", len(all_stories), "stories read.")
    return all_stories
