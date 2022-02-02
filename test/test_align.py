# Importing Dependencies
import pytest
from align import NeedlemanWunsch, read_fasta
import numpy as np

def test_nw_alignment():
    """
    Unit test for nw alg alignment. Here using
    test_seq1.fa and test_seq2.fa by asserting
    that both matrices have the same elements,
    and shape (np.array_equal docs).
    Testing for matrices M, gapA, and gapB.
    
    Gap open penalty = -10
    Gap extend penalty = -1
    """
    seq1, _ = read_fasta("./data/test_seq1.fa")
    seq2, _ = read_fasta("./data/test_seq2.fa")

    # penalties
    gap_open = -10
    gap_extend = -1

    # after solving by hand each matrix should be:
    match_matrix = np.array([[0, -np.inf, -np.inf, -np.inf, -np.inf],
                             [-np.inf, 5, -12, -12, -14],
                             [-np.inf, -11, 4, -1, -6],
                             [-np.inf, -13, -8, 5, 4]])

    gapA_matrix = np.array([[-10, -np.inf, -np.inf, -np.inf, -np.inf],
                            [-11, -22, -23, -24, -25],
                            [-12, -6, -17, -18, -19],
                            [-13, -7, -7, -12, -17]])

    gapB_matrix = np.array([[-10, -11, -12, -13, -14],
                            [-np.inf, -22, -6, -7, -8],
                            [-np.inf, -23, -17, -7, -8],
                            [-np.inf, -24, -18, -18, -6]])

    # needleman-wunsch with blosum62
    do_nw = NeedlemanWunsch("substitution_matrices/BLOSUM62.mat",
                         gap_open, gap_extend)
    do_nw.align(seq1, seq2)

    # asserting that all matrices have been filled correctly
    assert np.array_equal(do_nw._align_matrix, match_matrix), "Matrix M is incorrect :("
    assert np.array_equal(do_nw._gapA_matrix, gapA_matrix), "Matrix A is incorrect :( "
    assert np.array_equal(do_nw._gapB_matrix, gapB_matrix), "Matrix B is incorrect :("


def test_nw_backtrace():
    """
    Unit test for nw alg backtracing using new seqs
    test_seq3.fa and test_seq4.fa by asserting that
    the backtrace is correct. Here BLOSUM62 is used.
    Same gap open and gap extend penalty as above.
    """
    seq3, _ = read_fasta("./data/test_seq3.fa")
    seq4, _ = read_fasta("./data/test_seq4.fa")
    

    gap_open = -10
    gap_extend = -11

    # create obj based on BLOSUM62 matrix
    do_nw = NeedlemanWunsch("substitution_matrices/BLOSUM62.mat",
                         gap_open, gap_extend)
    # get the best scores
    al_score, seqA_align, seqB_align = do_nw.align(seq3, seq4)

    # assert the results are correct
    assert al_score == 17, "Alignment Score is incorrect :("
    assert seqA_align == 'MAVHWLIRRP', "Sequence 3 alignment is incorrect :("
    assert seqB_align == 'M---QLIRHP', "Sequence 4 alignment is incorrect :("




