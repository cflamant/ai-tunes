"""
Contains various tools to aid in preparing MIDI tracks for training the
neural network, as well as tools to convert outputted piano roll format
music into MIDI files to be played.
"""
import os


def get_files_list(dirName, recursive=True):
    """Get list of all MIDI files in directory.

    Returns a list of strings containing the full path of all MIDI
    files within the given directory. By default, the search is
    recursive, traversing the entire directory structure contained
    in the root.

    Parameters
    ----------
    dirName : str
        Absolute path of root directory to search in.
    recursive : bool, optional
        Whether to enter directories contained in the root directory
        while searching for MIDI file paths.

    Returns
    -------
    midipaths : list of str
        List of full paths to MIDI files found in given directory.

    """

    # Create a list of files and subdirectories in the root directory
    allfiles = os.listdir(dirName)
    midipaths = list()

    for entry in allfiles:
        fullpath = os.path.join(dirName, entry)
        # If entry is a directory, get the list of files in the directory
        if recursive and os.path.isdir(fullpath):
            midipaths = midipaths + get_files_list(fullpath,
                                                   recursive=recursive)
        else:
            if fullpath.lower().endswith('.mid'):
                midipaths.append(fullpath)

    return midipaths


def write_files_list(midipaths, filename):
    """Write list of MIDI paths to a file.

    Parameters
    ----------
    midipaths : list of str
        List of full paths to MIDI files.
    filename : str
        Filename or path to save file to.

    """

    with open(filename, 'w') as f:
        for item in midipaths:
            f.write("%s\n" % item)


def read_files_list(filename):
    """Read list of MIDI paths from file.

    Parameters
    ----------
    filename : str
        Filename or path of file with MIDI paths written one per line.

    Returns
    -------
    midipaths : list of str
        List of full paths to MIDI files.

    """

    midipaths = [line.rstrip('\n') for line in open(filename)]
    return midipaths
