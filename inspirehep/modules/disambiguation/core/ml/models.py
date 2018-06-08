# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2017 CERN.
#
# INSPIRE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# INSPIRE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.
#
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""Disambiguation core ML models."""

from __future__ import absolute_import, division, print_function

import csv
import json
import pickle

import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from beard.similarity import (
    CosineSimilarity,
    ElementMultiplication,
    EstimatorTransformer,
    PairTransformer,
    StringDistance,
)
from beard.utils import FuncTransformer, Shaper, normalize_name
from inspirehep.modules.disambiguation.core.ml.utils import (
    get_abstract,
    get_author_affiliation,
    get_author_full_name,
    get_author_other_names,
    get_coauthors_from_range,
    get_collaborations,
    get_first_given_name,
    get_keywords,
    get_second_given_name,
    get_second_initial,
    get_title,
    get_topics,
    group_by_signature,
)
from inspirehep.modules.disambiguation.utils import open_file_in_folder


class EthnicityEstimator(object):
    def __init__(self, C=4.0):
        self.C = C

    def load_data(self, input_filename):
        ethnicities, lasts, firsts = [], [], []
        with open(input_filename, 'r') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                ethnicities.append(int(row['RACE']))
                lasts.append(row['NAMELAST'])
                firsts.append(row['NAMEFRST'])

        names = ['%s, %s' % (last, first) for last, first in zip(lasts, firsts)]
        normalized_names = [normalize_name(name) for name in names]

        self.X = normalized_names
        self.y = ethnicities

    def load_model(self, input_filename):
        with open(input_filename, 'r') as fd:
            self.estimator = pickle.load(fd)

    def save_model(self, output_filename):
        with open_file_in_folder(output_filename, 'w') as fd:
            pickle.dump(self.estimator, fd, protocol=pickle.HIGHEST_PROTOCOL)

    def fit(self):
        self.estimator = Pipeline([
            ('transformer', TfidfVectorizer(analyzer='char_wb',
                                            ngram_range=(1, 5),
                                            min_df=0.00005,
                                            dtype=np.float32,
                                            decode_error='replace')),
            ('classifier', LinearSVC(C=self.C)),
        ])
        self.estimator.fit(self.X, self.y)

    def predict(self, X):
        return self.estimator.predict(X)


class DistanceEstimator(object):
    def __init__(self, ethnicity_model_path):
        self.ethnicity_estimator = EthnicityEstimator()
        self.ethnicity_estimator.load_model(ethnicity_model_path)

    def load_data(self, signatures_path, pairs_path, pairs_size, publications_path):
        reversed_publications = {}
        with open(publications_path, 'r') as fd:
            for line in fd:
                publication = json.loads(line)
                reversed_publications[publication['publication_id']] = publication

        reversed_signatures = {}
        with open(signatures_path, 'r') as fd:
            for line in fd:
                signature = json.loads(line)
                signature['publication'] = reversed_publications[signature['publication_id']]
                reversed_signatures[signature['signature_uuid']] = signature

        self.X = np.empty((pairs_size, 2), dtype=np.object)
        self.y = np.empty(pairs_size, dtype=np.int)

        with open(pairs_path, 'r') as fd:
            for i, line in enumerate(fd):
                pair = json.loads(line)
                self.X[i, 0] = reversed_signatures[pair['signature_uuids'][0]]
                self.X[i, 1] = reversed_signatures[pair['signature_uuids'][1]]
                self.y[i] = 0 if pair['same_cluster'] else 1

    def load_model(self, input_filename):
        with open(input_filename, 'r') as fd:
            self.distance_estimator = pickle.load(fd)

    def save_model(self, output_filename):
        with open_file_in_folder(output_filename, 'w') as fd:
            pickle.dump(self.distance_estimator, fd, protocol=pickle.HIGHEST_PROTOCOL)

    def fit(self):
        transformer = FeatureUnion([
            ('author_full_name_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('full_name', FuncTransformer(func=get_author_full_name)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(analyzer='char_wb',
                                               ngram_range=(2, 4),
                                               dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('author_second_initial_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=FuncTransformer(
                    func=get_second_initial
                ), groupby=group_by_signature)),
                ('combiner', StringDistance(
                    similarity_function='character_equality'))
            ])),
            ('author_first_given_name_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=FuncTransformer(
                    func=get_first_given_name
                ), groupby=group_by_signature)),
                ('combiner', StringDistance())
            ])),
            ('author_second_given_name_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=FuncTransformer(
                    func=get_second_given_name
                ), groupby=group_by_signature)),
                ('combiner', StringDistance())
            ])),
            ('author_other_names_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('other_names', FuncTransformer(
                        func=get_author_other_names)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(analyzer='char_wb',
                                               ngram_range=(2, 4),
                                               dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('affiliation_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('affiliation', FuncTransformer(
                        func=get_author_affiliation)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(analyzer='char_wb',
                                               ngram_range=(2, 4),
                                               dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('coauthors_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('coauthors', FuncTransformer(
                        func=get_coauthors_from_range)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('abstract_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('abstract', FuncTransformer(func=get_abstract)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('keywords_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('keywords', FuncTransformer(func=get_keywords)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('collaborations_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('collaborations', FuncTransformer(
                        func=get_collaborations)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('subject_similairty', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('keywords', FuncTransformer(func=get_topics)),
                    ('shaper', Shaper(newshape=(-1))),
                    ('tf-idf', TfidfVectorizer(dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('title_similarity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('title', FuncTransformer(func=get_title)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('tf-idf', TfidfVectorizer(analyzer='char_wb',
                                               ngram_range=(2, 4),
                                               dtype=np.float32,
                                               decode_error='replace')),
                ]), groupby=group_by_signature)),
                ('combiner', CosineSimilarity())
            ])),
            ('author_ethnicity', Pipeline([
                ('pairs', PairTransformer(element_transformer=Pipeline([
                    ('name', FuncTransformer(func=get_author_full_name)),
                    ('shaper', Shaper(newshape=(-1,))),
                    ('classifier', EstimatorTransformer(self.ethnicity_estimator.estimator)),
                ]), groupby=group_by_signature)),
                ('sigmoid', FuncTransformer(func=expit)),
                ('shaper', Shaper(newshape=(-1, 2))),
                ('combiner', ElementMultiplication())
            ]))
        ])
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=8)

        self.distance_estimator = Pipeline([('transformer', transformer), ('classifier', classifier)])
        self.distance_estimator.fit(self.X, self.y)

    def predict(self, X):
        return self.distance_estimator.predict(X)
