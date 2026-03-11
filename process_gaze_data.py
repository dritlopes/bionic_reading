import re
from typing import List
import pprint
from pathlib import Path
import pandas as pd
import json
import numpy as np

class InterestArea:

    def __init__(self,
                 word_id:int,
                 word:str,
                 sxp:float,
                 syp:float,
                 exp:float,
                 eyp:float,
                 line:int):

        self.word_id = word_id
        self.word = word
        self.sxp = sxp
        self.syp = syp
        self.exp = exp
        self.eyp = eyp
        self.line = line

    def to_dict(self) -> dict:
        return {'word_id': self.word_id,
                'word': self.word,
                'sxp': self.sxp,
                'syp': self.syp,
                'exp': self.exp,
                'eyp': self.eyp,
                'line_id': self.line}

class TextBlock:

    def __init__(self,
                 ias:List[InterestArea]=None,
                 text_id:str=None,
                 manipulation:str=None,
                 line_height:float=None,
                 midlines:List[float]=None,
                 screen_width:float=None,
                 screen_height:float=None,
                 font_size:float=None,
                 font_face:str=None):

        self.ias = ias
        self.text_id = text_id
        self.manipulation = manipulation
        self.line_height = line_height
        self.midlines = midlines
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font_size = font_size
        self.font_face = font_face

    def parse_text_into_ias(self, words_df:pd.DataFrame):


        y_end_of_lines = words_df['y_end_new'].unique()
        lines_ids = [i for i, y_end in enumerate(y_end_of_lines)]
        line_mapping = dict(zip(y_end_of_lines, lines_ids))

        ias = []
        for row in words_df.itertuples():
            ia = InterestArea(word_id = int(row.word_index),
                              word = str(row.word_name),
                              sxp = float(row.x_beginning_new),
                              syp = float(row.y_beginning_new),
                              exp = float(row.x_end_new),
                              eyp = float(row.y_end_new),
                              line = int(line_mapping[row.y_end_new]))
            ias.append(ia)
        self.ias = ias

    def find_line_height(self, words_df:pd.DataFrame):

        # compute difference between y_end of a line and y_start of the next line (= line height)
        y_end_of_lines = words_df['y_end'].unique()
        y_start_of_lines = words_df['y_beginning'].unique()
        line_height = abs(abs(y_end_of_lines[0]) - abs(y_start_of_lines[1]))
        self.line_height = line_height

    def find_midlines(self, words_df:pd.DataFrame):

        # compute y coordinate corresponding to the centre of each line
        midlines = []
        y_end_of_lines = words_df['y_end_new'].unique()
        y_start_of_lines = words_df['y_beginning_new'].unique()
        for y_end, y_start in zip(y_end_of_lines, y_start_of_lines):
            line_height = abs(abs(y_end - y_start))
            midline = line_height / 2
            y_midline = y_start + midline
            midlines.append(y_midline)
        self.midlines = midlines

class Sample:

    def __init__(self, time:int, xp:float, yp:float, ia:InterestArea|None=None):

        self.time = time
        self.xp = xp
        self.yp = yp
        self.ia = ia

    def to_dict(self):
        return {'time': self.time, 'xp': self.xp, 'yp': self.yp,
                'ia': None if self.ia is None else self.ia.to_dict()}

class Fixation:

    def __init__(self,
                 start_time:int,
                 end_time:int,
                 xp:float,
                 yp:float,
                 dur:int,
                 pupil_size:float,
                 eye_side:str,
                 samples:List[Sample]|None=None,
                 ia: InterestArea | None = None) -> None:

        self.event_type = "fixation"
        self.start_time = start_time
        self.end_time = end_time
        self.xp = xp
        self.yp = yp
        self.dur = dur
        self.pupil_size = pupil_size
        self.eye_side = eye_side
        self.samples = samples
        self.ia = ia

    def to_dict(self) -> dict:

        return {'event_type': self.event_type,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'xp': self.xp,
                'yp': self.yp,
                'dur': self.dur,
                'pupil_size': self.pupil_size,
                'eye_side': self.eye_side,
                'ia': None if self.ia is None else self.ia.to_dict(),
                'samples': None if self.samples is None else [sample.to_dict() for sample in self.samples if sample]}

class Saccade:

    def __init__(self,
                 start_time:int,
                 end_time:int,
                 dur:int,
                 sxp:float,
                 syp:float,
                 exp:float,
                 eyp:float,
                 ampl:float,
                 samples:List[Sample]|None=None):

        self.event_type = "saccade"
        self.start_time = start_time
        self.end_time = end_time
        self.dur = dur
        self.sxp = sxp
        self.syp = syp
        self.exp = exp
        self.eyp = eyp
        self.ampl = ampl
        self.samples = samples

    def to_dict(self) -> dict:
        return {'event_type': self.event_type,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'dur': self.dur,
                'sxp': self.sxp,
                'syp': self.syp,
                'exp': self.exp,
                'eyp': self.eyp,
                'ampl': self.ampl,
                'samples': None if self.samples is None else [sample.to_dict() for sample in self.samples if sample]
                }

class Trial:

    def __init__(self,
                 trial_id:str='',
                 text_id:str='',
                 manipulation:str='',
                 events:List[Fixation|Saccade]=None,
                 text:TextBlock|None=None) -> None:

        self.trial_id = trial_id
        self.text_id = text_id
        self.manipulation = manipulation
        self.events = events
        self.text = text

    def parse_trial_header(self, trial_block:List[str]):

        trial_id_re = re.compile(r'^MSG\s+\S+\s+Trial number:\s+', re.IGNORECASE)
        paragraph_id_re = re.compile(r'^MSG\s+\S+\s+Paragraph\scanvas_experiment_', re.IGNORECASE)
        manipulation_re = re.compile(r'^MSG\s+\S+\s+Manipulation:\s+', re.IGNORECASE)

        trial_id, paragraph_id, manipulation = None, None, None

        # get trial level info
        for line in trial_block:
            if trial_id_re.match(line):
                trial_id = str(trial_id_re.sub('', line).strip())
            if paragraph_id_re.match(line):
                paragraph_id = str(paragraph_id_re.sub('', line).strip())
            if manipulation_re.match(line):
                manipulation = manipulation_re.sub('', line).strip()
            if trial_id and paragraph_id and manipulation:
                break

        self.trial_id = trial_id
        self.text_id = paragraph_id
        self.manipulation = manipulation

    @staticmethod
    def _parse_event_blocks_from_asc(trial_block: List[str], verbose=False) -> List[List[str]]:

        """
        Group lines in trial into events. Each event corresponds to a fixation or a saccade. Each line within an event corresponds to a millisecond sample.
        :param trial_block: list of lines belonging to the same trial. The line contain info on the events identified in the trial.
        :param verbose: True to print the total number of fixations and saccades, and the first and last 5 lines of the first 3 events.
        :return: list of event blocks. Each event block is a list of strings. Each string corresponds to a line in the .asc file.
        """

        event_blocks: List[List[str]] = []
        current_block: List[str] | None = None

        # to ignore header of trial
        first_event_re = re.compile(r'^MSG\s+\S+\s+Start of paragraph presentation\.', re.IGNORECASE)
        # it will match lines starting with SSACC or SFIX
        start_event_re = re.compile(r'^SSACC\s[LR]\s+\d+|^SFIX\s[LR]\s+\d+')

        for i, line in enumerate(trial_block):
            if start_event_re.match(line):
                # finish the previous block
                if current_block is not None:
                    event_blocks.append(current_block)
                # start a fresh block
                current_block = [line]
            else:
                # add line to current block
                if current_block is not None:
                    current_block.append(line)
                # start first block
                elif first_event_re.match(trial_block[i - 1]):
                    current_block = [line]
        # append last block
        if current_block is not None: event_blocks.append(current_block)

        if verbose:
            n_fixations = len([line for event in event_blocks for line in event if re.compile(r'^EFIX.+').match(line)])
            n_saccades = len([line for event in event_blocks for line in event if re.compile(r'^ESACC.+').match(line)])
            print('\n' + trial_block[0])
            print(f"Found {n_fixations} fixations and {n_saccades} saccades")
            # show the first 3 lines of the first 3 blocks (just for demo)
            for i, block in enumerate(event_blocks[:3], 1):
                print(f"\n--- Event {i} (first and final 5 lines) ---")
                pprint.pprint(block[:5])
                pprint.pprint('...')
                pprint.pprint(block[-5:])

        return event_blocks

    @staticmethod
    def slice(fixation_locations:List[tuple],
              line_height:float,
              midlines:List[float],
              x_thresh:int=192, y_thresh:int=32, w_thresh:int=32, n_thresh:int=90):

        """
        Adapted from EyeKit.
        Form a set of runs (sequence of fixations that are located within a certain vertical and horizontal distance to each other are grouped together)
        and will be assigned to the same line), and then reduce the set to *m* by repeatedly merging those that appear to be on the same line.
        Merged sequences are then assigned to text lines in positional order.
        Original method by [Glandorf & Schroeder (2021)](https://doi.org/10.1016/j.procs.2021.09.069).
        :param fixation_locations: list of xy coordinates of each fixation in a tuple.
        :param line_height: Distance between lines of text in pixels (average vertical distance between lines). In general, for single line spacing, the line height is equal to the font size.
        :param midlines: Y-coordinate of the midline of each line of text. Each midline is the y position of the text midline.
        :param x_thresh: minimum x distance (in pixels) between two fixations to generate a new run.
        :param y_thresh: minimum y distance (in terms of the average line height) between two fixations to generate a new run.
        :param w_thresh: within-line threshold (in terms of the average line distance). If the average distance of a run is below this threshold, the run is merged into the current proto-line.
        :param n_thresh: next-line threshold (in terms of the average line distance). If the average distance of a run is larger than w_threshold but smaller than n_threshold, it will be assigned to a new proto-line.
        :return:
        """

        fixation_XY = np.array(fixation_locations, dtype=int)
        line_Y = np.array(midlines, dtype=int)
        proto_lines, phantom_proto_lines = {}, {}

        # 1. Segment runs
        # "The vertical and horizontal distance between subsequent fixations is calculated. Every time one of
        # the distances exceeds its respective threshold x_thresh or y_thresh, a new run is initiated."
        dist_X = abs(np.diff(fixation_XY[:, 0]))
        dist_Y = abs(np.diff(fixation_XY[:, 1]))
        end_run_indices = list(np.where(np.logical_or(dist_X > x_thresh, dist_Y > y_thresh))[0] + 1)
        run_starts = [0] + end_run_indices
        run_ends = end_run_indices + [len(fixation_XY)]
        runs = [list(range(start, end)) for start, end in zip(run_starts, run_ends)]

        # 2. Determine starting run
        # " The run with the largest horizontal span is used as the starting point for the following
        # grouping process."
        longest_run_i = np.argmax([fixation_XY[run[-1], 0] - fixation_XY[run[0], 0] for run in runs])
        proto_lines[0] = runs.pop(longest_run_i)

        # 3. Group runs into proto lines
        # "Runs are grouped together into proto-lines which will later be assigned to actual lines of the text.
        while runs:
            merger_on_this_iteration = False
            for proto_line_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
                # Create new proto line above or below (depending on direction)
                proto_lines[proto_line_i + direction] = []
                # Get current proto line XY coordinates (if proto line is empty, get phanton coordinates)
                if proto_lines[proto_line_i]:
                    proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
                else:
                    proto_line_XY = phantom_proto_lines[proto_line_i]
                # Compute differences between current proto line and all runs
                run_differences = np.zeros(len(runs))
                for run_i, run in enumerate(runs):
                    y_diffs = [
                        y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                        for x, y in fixation_XY[run]
                    ]
                    run_differences[run_i] = np.mean(y_diffs)
                # Find runs that can be merged into this proto line
                merge_into_current = list(np.where(abs(run_differences) < w_thresh)[0])
                # Find runs that can be merged into the adjacent proto line
                merge_into_adjacent = list(
                    np.where(
                        np.logical_and(
                            run_differences * direction >= w_thresh,
                            run_differences * direction < n_thresh,
                        )
                    )[0]
                )
                # Perform mergers
                for index in merge_into_current:
                    proto_lines[proto_line_i].extend(runs[index])
                for index in merge_into_adjacent:
                    proto_lines[proto_line_i + direction].extend(runs[index])
                # If no, mergers to the adjacent, create phantom line for the adjacent
                if not merge_into_adjacent:
                    average_x, average_y = np.mean(proto_line_XY, axis=0)
                    adjacent_y = average_y + line_height * direction
                    phantom_proto_lines[proto_line_i + direction] = np.array(
                        [[average_x, adjacent_y]]
                    )
                # Remove all runs that were merged on this iteration
                for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                    del runs[index]
                    merger_on_this_iteration = True
            # If no mergers were made, break the while loop
            if not merger_on_this_iteration:
                break

        # 4. Assign any leftover runs to the closest proto lines
        for run in runs:
            best_pl_distance = np.inf
            best_pl_assignemnt = None
            for proto_line_i in proto_lines:
                if proto_lines[proto_line_i]:
                    proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
                else:
                    proto_line_XY = phantom_proto_lines[proto_line_i]
                y_diffs = [
                    y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                    for x, y in fixation_XY[run]
                ]
                pl_distance = abs(np.mean(y_diffs))
                if pl_distance < best_pl_distance:
                    best_pl_distance = pl_distance
                    best_pl_assignemnt = proto_line_i
            proto_lines[best_pl_assignemnt].extend(run)

        # 5. Prune proto lines
        while len(proto_lines) > len(line_Y):
            top, bot = min(proto_lines), max(proto_lines)
            if len(proto_lines[top]) < len(proto_lines[bot]):
                proto_lines[top + 1].extend(proto_lines[top])
                del proto_lines[top]
            else:
                proto_lines[bot - 1].extend(proto_lines[bot])
                del proto_lines[bot]

        # 6. Map proto lines to text lines
        for line_i, proto_line_i in enumerate(sorted(proto_lines)):
            fixation_XY[proto_lines[proto_line_i], 1] = line_Y[line_i]

        return fixation_XY[:, 1]

    def snap_fixations_to_lines(self, fixation_locations, method='slice', **kwargs):

        pass
        # print([event.yp for event in self.events if event.event_type=='fixation'])

        # do line assignment (vertical drift correction)
        if method == 'slice':
            corrected_Y = self.slice(fixation_locations, **kwargs)
            # replace y coordinates with corrected y coordinates
            for event, y in zip(self.events, corrected_Y):
                if event.event_type=='fixation':
                    event.yp = y

        # print([event.yp for event in self.events if event.event_type=='fixation'])

    @ staticmethod
    def map_sample_to_word(ias, xp, yp):

        word = None
        if ias:
            for ia in ias:
                # check whether xp and yp are in the range of an IA corresponding to a word
                # rounding down the decimal pixels of the fixation!!!
                if ia.sxp <= int(xp) <= ia.exp and ia.syp <= int(yp) <= ia.eyp:
                    word = ia
                    break
        return word

    def map_samples_to_words(self, text, snap_to_lines, line_assignment_method):

        if snap_to_lines:
            fixation_locations = [(event.xp, event.yp) for event in self.events if event.event_type == 'fixation']
            if line_assignment_method == 'slice':
                self.snap_fixations_to_lines(fixation_locations, line_assignment_method, line_height=text.line_height, midlines=text.midlines)

        for event_i, event in enumerate(self.events):

            if event.event_type=='fixation':

                word = self.map_sample_to_word(text.ias, event.xp, event.yp)
                self.events[event_i].ia = word

                if event.samples:

                    for sample_i, sample in enumerate(event.samples):

                        word = self.map_sample_to_word(text.ias, sample.xp, sample.yp)
                        self.events[event_i].samples[sample_i].ia = word

    @staticmethod
    def parse_samples_from_event_block(event_block:List[str]) -> List[Sample]:

        sample_re = re.compile(r'^\d+\t.*')  # it will match any line starting with a timestamp, e.g. 2363391
        samples=[]

        for line in event_block:

            # timestamp,  xp,  yp  ...
            if sample_re.match(line):
                cells = [cell.strip() for cell in line.split('\t')]
                # do not include blink samples
                if cells[1] == '.' and cells[2] == '.':
                    continue
                else:
                    xp, yp = float(cells[1]), float(cells[2])
                    time = int(cells[0])
                    samples.append(Sample(time, xp, yp))

        return samples

    @staticmethod
    def parse_fixation(event_block:List[str]) -> Fixation:

        end_fix_sample_re = re.compile(r'^EFIX.+')  # it will match the lines starting with EFIX

        for line in event_block:

            if end_fix_sample_re.match(line):
                # EFIX L/R timestamp_begin, timestamp_end, dur, mean_xp, mean_yp ...
                cells = [cell.strip() for cell in line.split('\t')]
                xp, yp = float(cells[3]), float(cells[4])
                start_time = int(cells[0].split()[-1])
                end_time = int(cells[1])
                dur = int(cells[2])
                pupil_size = int(cells[5])
                eye_side = str(cells[0].split()[1])

        return Fixation(start_time, end_time, xp, yp, dur, pupil_size, eye_side)

    @staticmethod
    def parse_saccade(event_block:List[str]) -> Saccade:

        end_sacc_sample_re = re.compile(r'^ESACC.+')  # it will match the lines starting with ESACC

        for line in event_block:

            if end_sacc_sample_re.match(line):
                # EFIX L/R timestamp_begin, timestamp_end, dur, mean_xp, mean_yp ...
                cells = [cell.strip() for cell in line.split('\t')]
                start_time = int(cells[0].split()[-1])
                end_time = int(cells[1])
                dur = int(cells[2])
                sxp = float(cells[3])
                syp = float(cells[4])
                exp = float(cells[5])
                eyp = float(cells[6])
                ampl = float(cells[7])

        return Saccade(start_time, end_time, dur, sxp, syp, exp, eyp, ampl)

    def parse_events(self, trial_block:List[str], parse_samples:bool=True, verbose:bool=False):

        # slice event blocks from trial lines (a list of strings)
        event_blocks = self._parse_event_blocks_from_asc(trial_block, verbose)

        # identify fixations and saccades from event blocks
        events = []
        for event_block in event_blocks:
            if re.compile(r'^EFIX.+').match(event_block[-1]):
                fixation = self.parse_fixation(event_block)
                if parse_samples:
                    samples = self.parse_samples_from_event_block(event_block)
                    fixation.samples = samples
                events.append(fixation)
            elif re.compile(r'^ESACC.+').match(event_block[-1]):
                saccade = self.parse_saccade(event_block)
                if parse_samples:
                    samples = self.parse_samples_from_event_block(event_block)
                    saccade.samples = samples
                events.append(saccade)

        self.events = events

    def to_dict(self):
        return {'trial_id': self.trial_id, 'text_id': self.text_id, 'manipulation': self.manipulation,
                'events': [event.to_dict() for event in self.events]}

class TrialSequence:

    def __init__(self, asc_path:Path|str, participant_id:str='', trials:List[Trial]|None=None) -> None:

        self.asc_path = asc_path
        if not participant_id:
            participant_id = re.search(r'sub_\d+', asc_path).group()
        self.participant_id = participant_id
        self.trials = trials

    @staticmethod
    def _parse_trial_blocks_from_asc(asc_path:Path|str, verbose:bool=False) -> List[List[str]]:

        """
        Group lines in eye-tracker .asc file into trials. Each trial corresponds to the presentation of a paragraph.
        :param asc_path: path to eye-tracker .asc file (output of SR EDF-converter)
        :param verbose: True to print total number of trials, and first and last 5 lines of first 3 trials
        :return: list of trial blocks. Each trial block is a list of strings. Each string corresponds to a line in the .asc file.
        """

        with open(asc_path, 'r') as f:
            lines = f.readlines()

        # keep the text but drop newline
        lines = [line.rstrip("\n") for line in lines]
        # print(len(lines))
        # print(repr(lines[0]))

        # regular expression that matches the line that starts a trial block.
        # MSG <msg_number> Trial number: <trial_number>
        # e.g. "MSG 42 Trial number: 5"
        trial_start_re = re.compile(r'^MSG\s+\S+\s+Trial number:\s+\d+', re.IGNORECASE)
        trial_end_re = re.compile(r'^MSG\s+\S+\s+End of paragraph presentation\.', re.IGNORECASE)

        trial_blocks: List[List[str]] = []
        current_block: List[str] | None = None

        # for each trial, save lines as a list of strings
        # excluding question presentations
        for line in lines:
            if trial_start_re.match(line):
                # a new trial starts here
                current_block = [line]
            elif trial_end_re.match(line):
                # finish the previous block
                current_block.append(line)
                trial_blocks.append(current_block)
                current_block = None
            else:
                # add line to the current block if we have one
                if current_block is not None:
                    current_block.append(line)

        if verbose:
            # if correct, should be the same as the number of texts in experiment (excluding practice trials) = 80
            print(f"Found {len(trial_blocks)} trial block(s)")
            # show the first 3 lines of the first 3 blocks (just for demo)
            for i, block in enumerate(trial_blocks[:3], 1):
                print(f"\n--- Trial {i} (first and last 5 lines) ---")
                pprint.pprint(block[:5])
                print(f"...")
                pprint.pprint(block[-5:])
            print('\n\n\n')

        return trial_blocks

    def parse_trial_blocks(self, parse_samples:bool=True, verbose_trial:bool=False, verbose_event:bool=False) -> List[Trial]:

        trial_blocks = self._parse_trial_blocks_from_asc(self.asc_path, verbose_trial)

        all_trials = []

        for trial_block in trial_blocks:

            # parse trial object
            trial = Trial()
            trial.parse_trial_header(trial_block)

            # parse events (and optionally samples) from trial
            trial.parse_events(trial_block, parse_samples, verbose_event)

            all_trials.append(trial)

        self.trials = all_trials

    def map_events_to_ias(self, texts:List[TextBlock], snap_to_lines:bool=False, line_assignment_method:str='slice')-> List[Trial]:

        for trial_i, trial in enumerate(self.trials):

            # take text read in this trial
            if texts:
                for text in texts:
                    if str(text.text_id) == str(trial.text_id) and text.manipulation == trial.manipulation:
                        trial.text = text
                        break

            if trial.text:
                trial.map_samples_to_words(trial.text, snap_to_lines, line_assignment_method)

    def to_json(self, path:Path|str):

        all_trials = {'asc_path': self.asc_path, 'participant_id': self.participant_id, 'trials': [trial.to_dict() for trial in self.trials]}

        with open(path, 'w') as f:
            json.dump(all_trials, f, indent=4)

def adjust_ia_boxes(words_df: pd.DataFrame, path_to_save: str = '') -> pd.DataFrame:

    # words_df['x_beginning'] = words_df['x_beginning'] + 2
    # words_df['x_end'] = words_df['x_end'] + 2
    words_df['y_beginning'] = words_df['y_beginning'] - 5
    words_df['y_end'] = words_df['y_end'] + 5

    if path_to_save:
        words_df.to_csv(path_to_save, index=False)

    return words_df

def convert_xy_coordinates(words_df: pd.DataFrame, path_to_save: str = '') -> pd.DataFrame:

    # convert xy coordinates from opensesame (0,0 at centre) reference to eyelink reference (0,0 at top left)
    words_df['x_beginning_new'] = words_df['x_beginning'].apply(lambda x: x + 512)
    words_df['x_end_new'] = words_df['x_end'].apply(lambda x: x + 512)
    words_df['y_beginning_new'] = words_df['y_beginning'].apply(lambda x: x + 384)
    words_df['y_end_new'] = words_df['y_end'].apply(lambda x: x + 384)

    if path_to_save:
        words_df.to_csv(path_to_save, index=False)

    return words_df

def change_manipulation_names(words_df: pd.DataFrame, path_to_save: str = '') -> pd.DataFrame:

    map_manipulation_names = {'bionic_reading': 'block_bionic',
                              'normal_reading': 'block_normal',
                              'word_importance_reading': 'block_word_importance',
                              'part_of_speech_reading': 'block_part_of_speech'}

    words_df['text_manipulation'] = words_df['text_manipulation'].apply(lambda x: map_manipulation_names[x])

    if path_to_save:
        words_df.to_csv(path_to_save, index=False)

    return words_df

def create_word_dataframe(trial_sequence: TrialSequence) -> pd.DataFrame:

    data_cols = {'participant': [], # e.g. sub_0
                'trial_id': [], # e.g. 0
                'paragraph_id': [], # e.g. 0
                'manipulation': [], # e.g. bionic_reading
                'fix_id': [], # e.g. 0
                'timestamp': [], # e.g. (2408123,2408148)
                'xp': [], # e.g. -5.7 (in pixels)
                'yp': [], # e.g. 162.3 (in pixels)
                'dur': [], # e.g. 300 (in milliseconds)
                'word_id': [], # e.g. 0
                'word': [], # e.g. "Toen"
                'in_sacc_start': [], # (x,y) coordinates of start of incoming saccade (in pixels)
                'in_sacc_end': [], # (x,y) coordinates of end of incoming saccade (in pixels)
                'in_sacc_len': [], # in words
                'in_sacc_dur': [],  # in milliseconds
                'out_sacc_start': [], # (x,y) coordinates of start of outgoing saccade (in pixels)
                'out_sacc_end': [], # (x,y) coordinates of end of outgoing saccade (in pixels)
                'out_sacc_len': [], # in words
                'out_sacc_dur': [], # in milliseconds
                'line_change': [] # whether there was a line change from previous fixation to current fixation
                }

    for trial in trial_sequence.trials:

        counter = 0

        for i, event in enumerate(trial.events):

            if event.event_type == 'fixation':

                # fixation basic info
                data_cols['fix_id'].append(counter)
                data_cols['timestamp'].append((event.start_time, event.end_time))
                data_cols['xp'].append(event.xp)
                data_cols['yp'].append(event.yp)
                data_cols['dur'].append(event.dur)
                word_id, word = None,None
                if event.ia:
                    word_id = event.ia.word_id
                    word = event.ia.word
                data_cols['word_id'].append(word_id)
                data_cols['word'].append(word)
                counter += 1

                # previous and upcoming fixations
                previous_fixation, upcoming_fixation = None, None
                if i-2 in range(len(trial.events)):
                    previous_fixation = trial.events[i-2]
                    assert previous_fixation.event_type == 'fixation'
                if i+2 in range(len(trial.events)):
                    upcoming_fixation = trial.events[i+2]
                    assert upcoming_fixation.event_type == 'fixation'

                # line change (only if current fixation and previous fixation were on words)
                line_change = 0
                if previous_fixation:
                    if previous_fixation.ia and event.ia:
                        if previous_fixation.ia.line != event.ia.line:
                            line_change = 1
                data_cols['line_change'].append(line_change)

                # start, end, dur and length of incoming saccade
                start_in, end_in, dur_in, len_in = None, None, None, None
                if i-1 in range(len(trial.events)):
                    incoming_saccade = trial.events[i-1]
                    assert incoming_saccade.event_type == 'saccade'
                    start_in = (incoming_saccade.sxp, incoming_saccade.syp)
                    end_in = (incoming_saccade.exp, incoming_saccade.eyp)
                    dur_in = incoming_saccade.dur
                    # previous fixation to determine incoming saccade length in words,
                    # assuming that location of the previous fixation is the same as the location of saccade initiation
                    if previous_fixation:
                        if event.ia and previous_fixation.ia:
                            len_in = abs(event.ia.word_id - previous_fixation.ia.word_id)
                data_cols['in_sacc_start'].append(start_in)
                data_cols['in_sacc_end'].append(end_in)
                data_cols['in_sacc_dur'].append(dur_in)
                data_cols['in_sacc_len'].append(len_in)

                # start, end, dur and length of outgoing saccade
                start_out, end_out, dur_out, len_out = None, None, None, None
                if i + 1 in range(len(trial.events)):
                    assert trial.events[i + 1].event_type == 'saccade'
                    start_out = (trial.events[i + 1].sxp, trial.events[i + 1].syp)
                    end_out = (trial.events[i + 1].exp, trial.events[i + 1].eyp)
                    dur_out = trial.events[i + 1].dur
                    if upcoming_fixation:
                        if event.ia and upcoming_fixation.ia:
                            len_out = abs(event.ia.word_id - upcoming_fixation.ia.word_id)
                data_cols['out_sacc_start'].append(start_out)
                data_cols['out_sacc_end'].append(end_out)
                data_cols['out_sacc_dur'].append(dur_out)
                data_cols['out_sacc_len'].append(len_out)

                # macro info
                data_cols['participant'].append(trial_sequence.participant_id)
                data_cols['trial_id'].append(trial.trial_id)
                data_cols['paragraph_id'].append(trial.text_id)
                data_cols['manipulation'].append(trial.manipulation)

    data = pd.DataFrame(data_cols)

    return data

def find_letter_x_location(word, word_location):

    letter_pos_to_pixels = []
    # TODO check if it is okay to assume that each letter has the same width (same number of pixels)
    letter_pixels = (word_location[1] - word_location[0])/len(word)
    begin_location = word_location[0]

    for i in range(len(word)):
        letter_pos_to_pixels.append((begin_location, begin_location + letter_pixels))
        begin_location += letter_pixels

    return letter_pos_to_pixels

def compute_reading_measures(fixation_data:pd.DataFrame, word_data:pd.DataFrame) -> pd.DataFrame:

    reading_data_rows = []

    for text_info, text_data in fixation_data.groupby(['participant', 'paragraph_id', 'manipulation']):

        text_words = word_data[(word_data['paragraph'] == text_info[1]) & (word_data['text_manipulation'] == text_info[2])]

        # remove first fixation for each text for each participant
        text_data = text_data[text_data['fix_id'] > 0]

        # remove fixations outside IAs
        text_data = text_data[text_data['word_id'] >= 0]

        for word in text_words.itertuples():

            word_fix_data = text_data[(text_data['word_id']==word.word_index) & (text_data['word']==word.word_name)]

            # reading measures defined here https://link.springer.com/article/10.3758/s13428-017-0908-4/tables/2
            # and here https://www.sr-research.com/support/thread-336.html
            row_dict = {'participant': text_info[0],
                        'text_id': text_info[1],
                        'manipulation': text_info[2],
                        'word_id': word.word_index,
                        'word': word.word_name,
                        'first_fix_dur': None, # The duration of the first fixation on a target region, provided that the first fixation does not occur after fixations on words further along in the text.
                        'gaze_dur': None, # The total duration of all fixations in a target region until the eyes fixate a region of text that is either progressive or regressive to the target region, provided that the first fixation on the target region does not occur after any fixations on words further along in the text.
                        'total_reading_time': None, # The total duration of all fixations in a target region.
                        'skip': None, # An interest area is considered skipped if no fixation occurred in first-pass reading.
                        'reg_out': None, # Whether regression(s) was made from the current interest area to earlier interest areas (e.g., previous parts of the sentence) prior to leaving that interest area in a forward direction. 1 if a saccade exits the current interest area to a lower word id (to the left in English) before a later interest area was fixated; 0 if not.
                        'reg_in': None, # Whether the current interest area received at least one regression from later interest areas (e.g., later parts of the sentence). 1 if interest area was entered from a higher word id (from the right in English); 0 if not.
                        'n_fix': None, # Total fixations falling in the interest area
                        'in_sacc_len': None, # Length in words of first incoming saccade
                        'in_sacc_dur': None, # Duration in milliseconds of first incoming saccade
                        'out_sacc_len': None,  # Length in words of first outgoing saccade
                        'out_sacc_dur': None,  # Duration in milliseconds of first outgoing saccade
                        'line_change': None, # If first incoming saccade resulted in line change
                        'landing_pos': None
                        }

            # if any fixation on word
            if not word_fix_data.empty:

                # compute FFD, GD, SKIP, and REG_OUT
                current_word_id = word.word_index
                first_fix_id = word_fix_data['fix_id'].tolist()[0]
                previous_fixations = text_data[text_data['fix_id'] < first_fix_id]
                # if no previous fixations or previous fixations were on words positioned before the current word,
                # the word was not skipped in first-pass reading
                if previous_fixations.empty or all(previous_fixations['word_id'] < current_word_id):

                    row_dict['skip'] = 0
                    row_dict['first_fix_dur'] = word_fix_data['dur'].tolist()[0]

                    # map pixels to letters and determine which letter the fixation was at
                    fix_x_loc = word_fix_data['xp'].tolist()[0]
                    letter_x_locations = find_letter_x_location(word.word_name, (word.x_beginning_new, word.x_end_new))
                    for i, x_loc in enumerate(letter_x_locations):
                        if x_loc[0] <= fix_x_loc < x_loc[1]:
                            row_dict['landing_pos'] = i
                            break

                    # gather first reading fixations
                    fix_ids_first_pass = [first_fix_id]
                    current_fix_id = first_fix_id
                    for fixation in word_fix_data.itertuples():
                        while fixation.fix_id - 1 == current_fix_id:
                            fix_ids_first_pass.append(fixation.fix_id)
                            current_fix_id = fixation.fix_id
                    first_pass_fixations = word_fix_data[word_fix_data['fix_id'].isin(fix_ids_first_pass)]
                    row_dict['gaze_dur'] = first_pass_fixations['dur'].sum()

                    # compute outgoing regression
                    reg_out = 0
                    last_first_pass_fix_id = first_pass_fixations['fix_id'].tolist()[-1]
                    # if next fixation after first-pass fixation is on earlier word, reg_out is 1.
                    next_fixation = text_data[text_data['fix_id'] == last_first_pass_fix_id + 1]
                    if not next_fixation.empty and next_fixation['word_id'].tolist()[0] < current_word_id:
                        reg_out = 1
                    row_dict['reg_out'] = reg_out

                # if any previous fixation was on later word in text, current word has been skipped in first-pass reading.
                else:
                    row_dict['skip'] = 1

                # all passes to compute TRT, REG_IN, NFIX, SACC
                row_dict['total_reading_time'] = sum(word_fix_data['dur'].tolist())
                row_dict['n_fix'] = len(word_fix_data)
                row_dict['in_sacc_len'] = word_fix_data['in_sacc_len'].tolist()[0]
                row_dict['in_sacc_dur'] = word_fix_data['in_sacc_dur'].tolist()[0]
                row_dict['out_sacc_len'] = word_fix_data['out_sacc_len'].tolist()[0]
                row_dict['out_sacc_dur'] = word_fix_data['out_sacc_dur'].tolist()[0]
                row_dict['line_change'] = word_fix_data['line_change'].tolist()[0]
                reg_in = 0
                for fixation in word_fix_data.itertuples():
                    previous_fixation = text_data[text_data['fix_id'] == fixation.fix_id - 1]
                    if not previous_fixation.empty and previous_fixation['word_id'].tolist()[0] > fixation.word_id:
                        reg_in = 1
                row_dict['reg_in'] = reg_in

            # if no fixations on word, word has been skipped.
            else:
                row_dict['skip'] = 1

            reading_data_rows.append(row_dict)

    reading_df = pd.DataFrame(reading_data_rows)

    # filter out too short first fixations (<80ms)
    reading_df = reading_df[(reading_df['first_fix_dur'].isna()) | (reading_df['first_fix_dur'] >= 80)]

    # # filter out too long total reading times (per participant, per manipulation)
    threshold = []
    for group_id, data in reading_df.groupby(['participant', 'manipulation']):
        reading_times = data['total_reading_time'].tolist()
        reading_times = [value for value in reading_times if not np.isnan(value)]
        cutoff = np.percentile(reading_times, 99)
        threshold.append((group_id[0], group_id[1], cutoff))
        # print(group_id[0], group_id[1], cutoff)
    for participant, manipulation, cutoff in threshold:
        reading_df = reading_df[(reading_df['participant_id']==participant)
                                & (reading_df['manipulation']==manipulation)
                                & (reading_df['total_reading_time']>=cutoff)]

    # # filter out too long total reading times (per participant, per manipulation)
    # cleaned_rows = []
    # for group_id, data in reading_df.groupby(['participant', 'manipulation']):
    #     reading_times = data['total_reading_time'].values.astype(float)
    #     mask_not_nan = ~np.isnan(reading_times)
    #     rt_clean = reading_times[mask_not_nan]
    #     if len(rt_clean) > 1:
    #         z_scores = (rt_clean - np.mean(rt_clean)) / np.std(rt_clean)
    #         outlier_mask = np.zeros(len(reading_times), dtype=bool)
    #         outlier_mask[mask_not_nan] = np.abs(z_scores) > 3
    #         upper = np.mean(rt_clean) + 3 * np.std(rt_clean)
    #     else:
    #         outlier_mask = np.zeros(len(reading_times), dtype=bool)
    #         upper = None
    #     cleaned_rows.append(data.loc[~outlier_mask])
    #     n_removed = len(data) - len(data.loc[~outlier_mask])
    #     print('participant: ', group_id[0])
    #     print('manipulation: ', group_id[1])
    #     print('threshold: ', upper)
    #     print("rows removed:", n_removed)
    # reading_df = pd.concat(cleaned_rows, ignore_index=True)

    return reading_df

def main():

    # data with word coordinates in opensesame experiment screen, needed to map x,y coordinates from eye-tracker to ia words
    word_path = 'data/word_coordinates_subject_0.csv'
    words_df = pd.read_csv(word_path)
    # optionally adjust IA boxes around words
    # words_df = adjust_ia_boxes(words_df, path_to_save=word_path)
    # map xy coordinates from opensesame to xy coordinates from eyelink
    words_df = convert_xy_coordinates(words_df, path_to_save=word_path.replace('.csv', '_converted.csv'))
    words_df = change_manipulation_names(words_df, path_to_save=word_path.replace('.csv', '_converted.csv'))
    words_df = pd.read_csv('data/word_coordinates_subject_0_adjusted_converted.csv')

    # create a TextBlock with IAs for every text
    texts = []
    for text_info, data in words_df.groupby(['paragraph', 'text_manipulation']):
        text = TextBlock(text_id=str(text_info[0]), manipulation=text_info[1],
                         screen_width=1024, screen_height=768,
                         font_size=22, font_face='Serif')
        text.parse_text_into_ias(data)
        text.find_line_height(data)
        text.find_midlines(data)
        texts.append(text)

    # data with log output from eye-tracker, converted from edf to asc
    asc_path = 'data/sub_0.asc'

    # parse trials from acs file
    parse_samples = False
    trial_sequence = TrialSequence(asc_path)
    trial_sequence.parse_trial_blocks(parse_samples=parse_samples)

    # mapping fixation locations to words
    snap_to_lines = False
    trial_sequence.map_events_to_ias(texts=texts, snap_to_lines=snap_to_lines)

    # # # visualise fixations on texts
    # # for trial in trial_sequence.trials:
    # #     visualise_fixations_on_text(trial)

    # save trial info dict as json
    path_to_save_json = f'data/trials_{trial_sequence.participant_id}.json'
    if parse_samples:
        path_to_save_json = path_to_save_json.replace('.json', '_samples.json')
    if snap_to_lines:
        path_to_save_json = path_to_save_json.replace('.json', '_line_correction.json')
    trial_sequence.to_json(path_to_save_json)

    # create dataframe with info on fixations (each row is a fixation)
    word_fixation_df = create_word_dataframe(trial_sequence)
    word_fixation_df.to_csv('data/fixation_data.csv', index=False)

    # compute reading measures (each row is a word)
    word_fixation_df = pd.read_csv('data/fixation_data.csv')
    words_df = pd.read_csv('data/word_coordinates_subject_0_adjusted_converted.csv')
    reading_data = compute_reading_measures(word_fixation_df, words_df)
    reading_data.to_csv('data/reading_measures.csv', index=False)

if __name__ == '__main__':
    main()