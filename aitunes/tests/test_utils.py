"""Place tests for utils file here"""

import numpy as np
import math
import aitunes.utils as utils


def test_piano_roll_to_instrument():
    "Ensure conversion of piano roll to instrument works"
    piano_roll = np.zeros((128, 5), dtype=int)
    # Play three notes that are held
    piano_roll[77, 0:2] = 100  # Note 1
    piano_roll[77, 3:] = 100   # Note 2
    piano_roll[60, :] = 100    # Note 3

    fs = 100
    program = 4
    inst = utils.piano_roll_to_instrument(piano_roll, fs=fs, program=program)
    assert inst.program == program
    note1 = False
    note2 = False
    note3 = False
    extranote = False
    for note in inst.notes:
        if note.pitch == 77:
            if (math.isclose(note.start, 0., rel_tol=1e-6) and
                    math.isclose(note.end, 0.02, rel_tol=1e-6) and
                    not note1):
                note1 = True
            elif (math.isclose(note.start, 0.03, rel_tol=1e-6) and
                    math.isclose(note.end, 0.05, rel_tol=1e-6) and
                    not note2):
                note2 = True
            else:
                extranote = True
        elif note.pitch == 60:
            if (math.isclose(note.start, 0., rel_tol=1e-6) and
                    math.isclose(note.end, 0.05, rel_tol=1e-6) and
                    not note3):
                note3 = True
        else:
            extranote = True
    assert note1 and note2 and note3 and not extranote
