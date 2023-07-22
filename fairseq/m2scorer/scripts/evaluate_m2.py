#!/usr/bin/python

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from m2scorer.scripts import levenshtein_py37
from m2scorer.scripts.m2scorer_py37 import load_annotation


def eval_m2(system_sentences, m2_file):
    max_unchanged_words=2
    beta = 0.5
    ignore_whitespace_casing= False
    verbose = False
    very_verbose = False

    # load source sentences and gold edits
    source_sentences, gold_edits = load_annotation(m2_file)
    p, r, fscore = levenshtein_py37.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose)
    return {'precision':p, 'recall':r, 'f0.5':fscore}


    