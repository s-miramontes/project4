# Importing Dependencies
import numpy as np
from typing import Tuple

# Defining class for Needleman-Wunsch Algorithm for Global pairwise alignment
class NeedlemanWunsch:
    """ Class for NeedlemanWunsch Alignment

    Parameters:
        sub_matrix_file: str
            Path/filename of substitution matrix
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty

    Attributes:
        seqA_align: str
            seqA alignment
        seqB_align: str
            seqB alignment
        alignment_score: float
            Score of alignment from algorithm
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty
    """
    def __init__(self, sub_matrix_file: str, gap_open: float, gap_extend: float):
        # Init alignment and gap matrices
        self._align_matrix = None 
        self._gapA_matrix = None 
        self._gapB_matrix = None 

        # Init matrices for backtrace procedure
        self._back = None 
        self._back_A = None 
        self._back_B = None 

        # Init alignment_score
        self.alignment_score = 0

        # Init empty alignment attributes
        self.seqA_align = ""
        self.seqB_align = ""

        # Init empty sequences
        self._seqA = ""
        self._seqB = ""

        # Setting gap open and gap extension penalties
        self.gap_open = gap_open
        assert gap_open < 0, "Gap opening penalty must be negative."
        self.gap_extend = gap_extend
        assert gap_extend < 0, "Gap extension penalty must be negative."

        # Generating substitution matrix
        self.sub_dict = self._read_sub_matrix(sub_matrix_file) # substitution dictionary

    def _read_sub_matrix(self, sub_matrix_file):
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This function reads in a scoring matrix from any matrix like file.
        Where there is a line of the residues followed by substitution matrix.
        This file also saves the alphabet list attribute.

        Parameters:
            sub_matrix_file: str
                Name (and associated path if not in current working directory)
                of the matrix file that contains the scoring matrix.

        Returns:
            dict_sub: dict
                Substitution matrix dictionary with tuple of the two residues as
                the key and score as value e.g. {('A', 'A'): 4} or {('A', 'D'): -8}
        """
        with open(sub_matrix_file, 'r') as f:
            dict_sub = {}  # Dictionary for storing scores from sub matrix
            residue_list = []  # For storing residue list
            start = False  # trigger for reading in score values
            res_2 = 0  # used for generating substitution matrix
            # reading file line by line
            for line_num, line in enumerate(f):
                # Reading in residue list
                if '#' not in line.strip() and start is False:
                    residue_list = [k for k in line.strip().upper().split(' ') if k != '']
                    start = True
                # Generating substitution scoring dictionary
                elif start is True and res_2 < len(residue_list):
                    line = [k for k in line.strip().split(' ') if k != '']
                    # reading in line by line to create substitution dictionary
                    assert len(residue_list) == len(line), "Score line should be same length as residue list"
                    for res_1 in range(len(line)):
                        dict_sub[(residue_list[res_1], residue_list[res_2])] = float(line[res_1])
                    res_2 += 1
                elif start is True and res_2 == len(residue_list):
                    break
        return dict_sub

    def align(self, seqA: str, seqB: str) -> Tuple[float, str, str]:
        """
        Performs Needleman-Wunsch Algorithm to align seqA, and seqB.
        Initialization of matrix with base cases occurs first.
        Filling in the rest of the matrices (M, A, B) based on the
        substrcture problem results.
        Once maximum value is obtained at each case, we fill
        backtracing matrices, while keeping track of the instance
        in which the maximum value was obtained.
        In other words, whether it was due a match (idx=0),
        a gap in matrix A (idx=1), or a gap in matrix B (idx=2)

        Inputs
        -----------
            self.seqA: str to align
            self.seqB: str to align

        Outputs
        ----------
            [from _backtrace()]
            self.alignment_score: best alignment score (bottom right)
            self._seqA_aligned: best aligned seqA
            self._seqB_aligned: best aligned seqB
       
        """
        # Initialize 6 matrix private attributes for use in alignment
        # create matrices for alignment scores and gaps
        self._align_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        self._gapA_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        self._gapB_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf

        # create matrices for pointers used in backtrace procedure
        self._back = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        self._back_A = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        self._back_B = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf

        # Resetting alignment in case method is called more than once (buffer)
        self.seqA_align = ""
        self.seqB_align = ""

        # Resetting alignment score in case method is called more than once
        self.alignment_score = 0

        # Initializing sequences for use in backtrace method
        self._seqA = seqA 
        self._seqB = seqB 


        # ---------------------------------- INITIALIZE FILLING -------------------------------
        # 1. Base case matrix M(atching)
        self._align_matrix[0,0] = 0 

        # 2. Base case matrix A(gap_open)
        # fill all rows in first col now
        for row in range(len(seqA) + 1):
            self._gapB_matrix[row, 0] = self.gap_open + self.gap_extend * row

        # 3. Base case matrix B(gap_extend)
        # fill first row, all cols now
        for col in range(len(seqB) + 1):
            self._gapA_matrix[0, col] = self.gap_open + self.gap_extend * col

        #---------------------------------------------------------------------------------Done.

        # -------------------------------- FILL A,B,M WITH DP --------------------------------
        # traveling down and to the left, again skipping base cases [0,0] 
        for i in range(1, len(seqA) + 1):

            for j in range(1, len(seqB) + 1):

                # Calculate max M[atch] score
                dict_match = self.sub_dict[(self._seqA[i-1]),
                                            (self._seqB[j-1])]
                match_max = [self._align_matrix[i-1, j-1] + dict_match,
                                self._gapA_matrix[i-1, j-1] + dict_match,
                                self._gapB_matrix[i-1, j-1] + dict_match]

                # asign M[i,j] to max match value
                self._align_matrix[i,j] = max(match_max)
                # save index of max value for backtracing 
                self._back[i,j] = np.argmax(match_max)


                # Calculate max gap in A gap score
                gapA_max = [self.gap_open + self.gap_extend + self._align_matrix[i,j-1],
                               self.gap_extend + self._gapA_matrix[i, j-1],
                               self.gap_open + self.gap_extend + self._gapB_matrix[i, j-1]]

                # assign A[i,j] to max gap_in_A value
                self._gapA_matrix[i,j] = max(gapA_max)
                # save index of max value for backtracing
                self._back_A[i,j] = np.argmax(gapA_max)


                # Calculate max gap in B gap score
                gapB_max = [self.gap_open + self.gap_extend + self._align_matrix[i-1, j],
                               self.gap_open + self.gap_extend + self._gapA_matrix[i-1, j],
                               self.gap_extend + self._gapB_matrix[i-1, j]]

                # assign B[i,j] to max gap_in_B value
                self._gapB_matrix[i,j] = max(gapB_max)
                # save index of max value for backtracing
                self._back_B[i,j] = np.argmax(gapB_max)
        #
        #        
        # final note:
        # we've kept track of the idx with the max score in each matrix cell via np.argmax
        # we'll use to our advantage in the _backtrace method.
        #
        # -------------------------------------------------------------------------------Done.

        return self._backtrace()

    def _backtrace(self) -> Tuple[float, str, str]:
        """

        This function performs the traceback path to 1) find the best alignment
        score, and 2) find the alignment of seqA and seqB that matches the 
        highest score.
        The heuristic chooses Matching over gaps (high road).

        Using backtracing matrices: _back, _back_A, and _back_B, the max score
        is selected based on the np.argmax index value stored.
        Visiting wither backtracing matrix will trace the path back to
        [0,0]. 
        This is further explained here:

            if np.argmax == 0:
                match, keep same letters in either seq.
            
            elif np.argmax == 1:
                gap in A, insert gap '-' in seqA
            
            else: (np.argmax ==2):
                gap in B, insert gap '-' in seqB
        """

        # Implement this method based upon the heuristic chosen in the align method above.
        lastRow = len(self._seqA)
        lastCol = len(self._seqB)

        # get best score (list score order, same as above)
        final_scores= [self._align_matrix[-1, -1],
                self._gapA_matrix[-1, -1], 
                self._gapB_matrix[-1, -1]]

        self.alignment_score = max(final_scores)

        # idx of best score (match, gap in A, or gap in B)
        max_idx = np.argmax(final_scores)


        while lastRow>0 and lastCol>0:

            # match was max score
            if max_idx == 0:
                # fill aligned sequences appropriately (with matching letters)
                self.seqA_align = self._seqA[lastRow - 1] + self.seqA_align
                self.seqB_align = self._seqB[lastCol - 1] + self.seqB_align

                # new best idx
                max_idx = self._back[lastRow, lastCol]

                # move diagonally (up to the left)
                lastRow -= 1
                lastCol -= 1


            # gapA was max score
            elif max_idx == 1:
                # fill aligned sequences appropriately (with gap in A)
                self.seqA_align = "-" + self.seqA_align
                self.seqB_align = self._seqB[lastCol - 1] + self.seqB_align

                # new best idx
                max_idx = self._back_A[lastRow, lastCol]

                # move left
                lastCol -= 1


            # gapB was max score
            else:
                # fill aligned sequences appropriately (with gap in B)
                self.seqA_align = self._seqA[lastRow - 1] + self.seqA_align
                self.seqB_align = "-" + self.seqB_align

                # new best idx
                max_idx = self._back_B[lastRow, lastCol]

                # move up
                lastRow -= 1

        # (finally) return tuple of alignment score, seqA aligned, seqB aligned
        return self.alignment_score, self.seqA_align, self.seqB_align


def read_fasta(fasta_file: str) -> Tuple[str, str]:
    """
    DO NOT MODIFY THIS FUNCTION! IT IS ALREADY COMPLETE!

    This function reads in a FASTA file and returns the associated
    string of characters (residues or nucleotides) and the header.
    This function assumes a single protein or nucleotide sequence
    per fasta file and will only read in the first sequence in the
    file if multiple are provided.

    Parameters:
        fasta_file: str
            name (and associated path if not in current working directory)
            of the Fasta file.

    Returns:
        seq: str
            String of characters from FASTA file
        header: str
            Fasta header
    """
    assert fasta_file.endswith(".fa"), "Fasta file must be a fasta file with the suffix .fa"
    with open(fasta_file) as f:
        seq = ""  # initializing sequence
        first_header = True
        for line in f:
            is_header = line.strip().startswith(">")
            # Reading in the first header
            if is_header and first_header:
                header = line.strip()  # reading in fasta header
                first_header = False
            # Reading in the sequence line by line
            elif not is_header:
                seq += line.strip().upper()  # generating full sequence
            # Breaking if more than one header is provided in the fasta file
            elif is_header and not first_header:
                break
    return seq, header
