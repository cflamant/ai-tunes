"""
Contains various tools to aid in preparing MIDI tracks for training the
neural network, as well as tools to convert outputted piano roll format
music into MIDI files to be played.
"""
import os
import numpy as np
import pretty_midi as pm
from mido.midifiles.meta import KeySignatureError
import warnings
from tqdm import tqdm


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
            f.write(f"{item}\n")


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


def prune_files_list(midipaths, min_len=10., max_len=1000., min_insts=1,
                     ignorewarnings=True, verbose=1):
    """Go through MIDI files specified by list of paths and remove
    any path whose file either does not load properly in pretty_midi,
    or does not meet the minimum length or number of instrument
    requirements specified.

    Parameters
    ----------
    midipaths : list of str
        List of full paths to MIDI files.
    min_len : float
        Minimum required length of MIDI file in seconds
    max_len : float
        Maxmimum allowed length in seconds. Useful to prevent memory
        issues later on when generating piano rolls if the file is
        corrupt or nonsensically long.
    min_insts : int
        Minimum number of instruments required in track
    ignorewarnings : bool
        Whether to ignore warnings when determining the validity
        of MIDI file. Usually raised by pretty_midi if the MIDI file
        might not be a valid type 0 or type 1. Default is True since
        it seems that most files raising this error are not strongly
        misinterpreted.
    verbose : int
        Whether to print out which file paths are not valid. 0 is for
        no output, 1 is for just printing the file path of offending
        files, and 2 is for printing the file path and the reason.
        1 and 2 print how many files are kept.

    Returns
    -------
    pruned_midipaths : list of str
        List of full paths to validated and pruned MIDI files.

    """
    pruned_midipaths = []
    numpaths = len(midipaths)
    if ignorewarnings:
        warnings.filterwarnings('ignore')
    else:
        warnings.filterwarnings('error')
    for midipath in tqdm(midipaths):
        try:
            md = pm.PrettyMIDI(midipath)
        except (KeySignatureError, IOError, EOFError, ValueError,
                IndexError, ZeroDivisionError, Exception, Warning) as e:
            if verbose:
                print(midipath)
                if verbose > 1:
                    print(e)
        else:
            numinst = len(md.instruments)
            endtime = md.get_end_time()
            if (numinst >= min_insts and
                    min_len <= endtime <= max_len):
                pruned_midipaths.append(midipath)
            else:
                if verbose:
                    print(midipath)
                    if verbose > 1:
                        print("Not enough instruments, or length unaccepted.")
                        print(f"Length: {endtime:.2f} s, ",
                              f"Number of Instruments: {numinst}")
    if verbose > 0:
        print(f"{len(pruned_midipaths)} of {numpaths} MIDI paths kept.")
    return pruned_midipaths


def piano_roll_to_instrument(piano_roll, fs=100, program=0):
    """Converts a Piano Roll array into notes added to a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument
    Returns
    -------
    inst : pretty_midi.Instrument
        A pretty_midi.Instrument class instance with notes corresponding to
        the piano roll.

    """

    npitch = piano_roll.shape[0]  # Expect 128
    inst = pm.Instrument(program=program)

    for i in range(npitch):
        # Get indices of nonzero elements
        idxs = np.nonzero(piano_roll[i, :])[0]
        j = 0
        # whether that note pitch is currently playing
        playing = False
        start = 0.
        pitch = i
        while j < idxs.shape[0]:
            # If not currently playing, start a new note
            if not playing:
                stime = idxs[j]
                start = stime*(1./fs)
                playing = True
                j += 1
                stime += 1
            elif idxs[j] != stime:  # Terminate previous note, restart loop
                playing = False
                # Note ended after previous nonzero frame
                end = (idxs[j-1]+1)*(1./fs)
                note = pm.Note(velocity=100, pitch=pitch, start=start, end=end)
                inst.notes.append(note)
            else:
                # Note is being held; increment curr. time and continue
                stime += 1
                j += 1
        if playing:
            # If still playing at the end of the piano roll, terminate the note
            end = (idxs[j-1]+1)*(1./fs)
            note = pm.Note(velocity=100, pitch=pitch, start=start, end=end)
            inst.notes.append(note)
    return inst


def analyze_tracks(midipaths, verbose=0):
    """Produces statistics on the MIDI files given in the list of
    paths to the songs.

    Parameters
    ----------
    midipaths : list of str
        List of full paths to MIDI files.
    verbose : int
        0 suppresses all print statements. 1 or higher prints when
        invalid MIDI files are encountered.

    Returns
    -------
    none # TODO

    """
    inst_nums = []
    song_lengths = []
    inst_types = []
    drum_types = []
    pitches = []
    numpaths = len(midipaths)
    print(numpaths)  # TODO remove

    for i, midipath in tqdm(enumerate(midipaths)):
        try:
            md = pm.PrettyMIDI(midipath)
        except (KeySignatureError, IOError, EOFError, ValueError,
                IndexError, ZeroDivisionError, Exception) as e:
            if verbose:
                print(midipath + " is invalid.")
                print(e)
        else:
            inst_nums.append(len(md.instruments))
            song_lengths.append(md.get_end_time())
            for inst in md.instruments:
                if not inst.is_drum:
                    inst_types.append(inst.program)
                    for note in inst.notes:
                        pitches.append(note.pitch)
                else:
                    for note in inst.notes:
                        drum_types.append(note.pitch)
    inst_nums = np.array(inst_nums)
    song_lengths = np.array(song_lengths)
    inst_types = np.array(inst_types)
    drum_types = np.array(drum_types)
    pitches = np.array(pitches)
    return inst_nums, song_lengths, inst_types, drum_types, pitches
