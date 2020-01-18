# -*- coding: utf-8 -*-
"""
Created on Tues Feb 22 09:30:44 2016

@author: Timothy Brathwaite
         José Ángel Martín Baos
@notes:  Credit is due to Akshay Vij and John Canny for the idea of using
         "mapping" matrices to avoid the need for "for loops" when computing
         quantities of interest such as probabilities, log-likelihoods,
         gradients, and hessians. This code is based on an earlier multinomial
         logit implementation by Akshay Vij which made use of such mappings.
         This version differs from version 1 by partitioning the parameters to
         be estimated, theta, as shape parameters, intercept parameters, and
         index coefficients.
"""
from __future__ import absolute_import

import warnings
import pickle
from copy import deepcopy
from functools import reduce
from numbers import Number
import random

import scipy.linalg
import scipy.stats
import numpy as np
import pandas as pd

from .choice_tools import create_design_matrix
from .choice_tools import create_long_form_mappings
from .choice_tools import convert_mixing_names_to_positions
from .choice_tools import get_dataframe_from_data
from .choice_tools import ensure_specification_cols_are_in_dataframe
from .choice_tools import ensure_object_is_ordered_dict
from .choice_tools import ensure_columns_are_in_dataframe
from .choice_calcs import calc_probabilities, calc_asymptotic_covariance
from .nested_choice_calcs import calc_nested_probs
from .nested_choice_calcs import naturalize_nest_coefs
from . import mixed_logit_calcs as mlc

# Create a list of the necesssary result dictionary keys
# that are expected from the estimation routines
needed_result_keys = ["final_log_likelihood",
                      "chosen_probs",
                      "long_probs",
                      "residuals",
                      "ind_chi_squareds",
                      "success",
                      "message",
                      "rho_squared",
                      "rho_bar_squared",
                      "log_likelihood_null",
                      "utility_coefs",
                      "intercept_params",
                      "shape_params",
                      "nest_params",
                      "final_gradient",
                      "final_hessian",
                      "fisher_info"]


def ensure_valid_nums_in_specification_cols(specification, dataframe):
    """
    Checks whether each column in `specification` contains numeric data,
    excluding positive or negative infinity and excluding NaN. Raises
    ValueError if any of the columns do not meet these requirements.

    Parameters
    ----------
    specification : iterable of column headers in `dataframe`.
    dataframe : pandas DataFrame.
        Dataframe containing the data for the choice model to be estimated.

    Returns
    -------
    None.
    """
    problem_cols = []
    for col in specification:
        # The condition below checks for values that are not floats or integers
        # This will catch values that are strings.
        if dataframe[col].dtype.kind not in ['f', 'i', 'u']:
            problem_cols.append(col)
        # The condition below checks for positive or negative inifinity values.
        elif np.isinf(dataframe[col]).any():
            problem_cols.append(col)
        # This condition will check for NaN values.
        elif np.isnan(dataframe[col]).any():
            problem_cols.append(col)

    if problem_cols != []:
        msg = "The following columns contain either +/- inifinity values, "
        msg_2 = "NaN values, or values that are not real numbers "
        msg_3 = "(e.g. strings):\n{}"
        total_msg = msg + msg_2 + msg_3
        raise ValueError(total_msg.format(problem_cols))

    return None


def ensure_ref_position_is_valid(ref_position, num_alts, param_title):
    """
    Ensures that `ref_position` is None or an integer that is in the interval
    `[0, num_alts - 1]`. If None, ensures that intercepts are not the
    parameters being estimated. Raises a helpful ValueError if otherwise.

    Parameters
    ----------
    ref_position : int.
        An integer denoting the position in an array of parameters that will
        be constrained for identification purposes.
    num_alts : int.
        An integer denoting the total number of alternatives in one's universal
        choice set.
    param_title : {'intercept_names', 'shape_names'}.
        String denoting the name of the parameters that are being estimated,
        with a constraint for identification. E.g. 'intercept_names'.

    Returns
    -------
    None.
    """
    assert param_title in ['intercept_names', 'shape_names']

    try:
        assert ref_position is None or isinstance(ref_position, int)
    except AssertionError:
        msg = "ref_position for {} must be an int or None."
        raise TypeError(msg.format(param_title))

    if param_title == "intercept_names":
        try:
            assert ref_position is not None
        except AssertionError:
            raise ValueError("At least one intercept should be constrained.")

    try:
        if ref_position is not None:
            assert ref_position >= 0 and ref_position <= num_alts - 1
    except AssertionError:
        msg = "ref_position must be between 0 and num_alts - 1."
        raise ValueError(msg)

    return None


def check_length_of_shape_or_intercept_names(name_list,
                                             num_alts,
                                             constrained_param,
                                             list_title):
    """
    Ensures that the length of the parameter names matches the number of
    parameters that will be estimated. Will raise a ValueError otherwise.

    Parameters
    ----------
    name_list : list of strings.
        Each element should be the name of a parameter that is to be estimated.
    num_alts : int.
        Should be the total number of alternatives in the universal choice set
        for this dataset.
    constrainted_param : {0, 1, True, False}
        Indicates whether (1 or True) or not (0 or False) one of the type of
        parameters being estimated will be constrained. For instance,
        constraining one of the intercepts.
    list_title : str.
        Should specify the type of parameters whose names are being checked.
        Examples include 'intercept_params' or 'shape_params'.

    Returns
    -------
    None.
    """
    if len(name_list) != (num_alts - constrained_param):
        msg_1 = "{} is of the wrong length:".format(list_title)
        msg_2 = "len({}) == {}".format(list_title, len(name_list))
        correct_length = num_alts - constrained_param
        msg_3 = "The correct length is: {}".format(correct_length)
        total_msg = "\n".join([msg_1, msg_2, msg_3])
        raise ValueError(total_msg)

    return None


def check_type_of_nest_spec_keys_and_values(nest_spec):
    """
    Ensures that the keys and values of `nest_spec` are strings and lists.
    Raises a helpful ValueError if they are.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.

    Returns
    -------
    None.
    """
    try:
        assert all([isinstance(k, str) for k in nest_spec])
        assert all([isinstance(nest_spec[k], list) for k in nest_spec])
    except AssertionError:
        msg = "All nest_spec keys/values must be strings/lists."
        raise TypeError(msg)

    return None


def check_for_empty_nests_in_nest_spec(nest_spec):
    """
    Ensures that the values of `nest_spec` are not empty lists.
    Raises a helpful ValueError if they are.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.

    Returns
    -------
    None.
    """
    empty_nests = []
    for k in nest_spec:
        if len(nest_spec[k]) == 0:
            empty_nests.append(k)
    if empty_nests != []:
        msg = "The following nests are INCORRECTLY empty: {}"
        raise ValueError(msg.format(empty_nests))

    return None


def ensure_alt_ids_in_nest_spec_are_ints(nest_spec, list_elements):
    """
    Ensures that the alternative id's in `nest_spec` are integers. Raises a
    helpful ValueError if they are not.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    list_elements : list of lists of ints.
        Each element should correspond to one of the alternatives identified as
        belonging to a nest.

    Returns
    -------
    None.
    """
    try:
        assert all([isinstance(x, int) for x in list_elements])
    except AssertionError:
        msg = "All elements of the nest_spec values should be integers"
        raise ValueError(msg)

    return None


def ensure_alt_ids_are_only_in_one_nest(nest_spec, list_elements):
    """
    Ensures that the alternative id's in `nest_spec` are only associated with
    a single nest. Raises a helpful ValueError if they are not.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    list_elements : list of ints.
        Each element should correspond to one of the alternatives identified as
        belonging to a nest.

    Returns
    -------
    None.
    """
    try:
        assert len(set(list_elements)) == len(list_elements)
    except AssertionError:
        msg = "Each alternative id should only be in a single nest."
        raise ValueError(msg)

    return None


def ensure_all_alt_ids_have_a_nest(nest_spec, list_elements, all_ids):
    """
    Ensures that the alternative id's in `nest_spec` are all associated with
    a nest. Raises a helpful ValueError if they are not.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    list_elements : list of ints.
        Each element should correspond to one of the alternatives identified as
        belonging to a nest.
    all_ids : list of ints.
        Each element should correspond to one of the alternatives that is
        present in the universal choice set for this model.

    Returns
    -------
    None.
    """
    unaccounted_alt_ids = []
    for alt_id in all_ids:
        if alt_id not in list_elements:
            unaccounted_alt_ids.append(alt_id)
    if unaccounted_alt_ids != []:
        msg = "Associate the following alternative ids with a nest: {}"
        raise ValueError(msg.format(unaccounted_alt_ids))

    return None


def ensure_nest_alts_are_valid_alts(nest_spec, list_elements, all_ids):
    """
    Ensures that the alternative id's in `nest_spec` are all in the universal
    choice set for this dataset. Raises a helpful ValueError if they are not.

    Parameters
    ----------
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    list_elements : list of ints.
        Each element should correspond to one of the alternatives identified as
        belonging to a nest.
    all_ids : list of ints.
        Each element should correspond to one of the alternatives that is
        present in the universal choice set for this model.

    Returns
    -------
    None.
    """
    invalid_alt_ids = []
    for x in list_elements:
        if x not in all_ids:
            invalid_alt_ids.append(x)
    if invalid_alt_ids != []:
        msg = "The following elements are not in df[alt_id_col]: {}"
        raise ValueError(msg.format(invalid_alt_ids))

    return None


def add_intercept_to_dataframe(specification, dataframe):
    """
    Checks whether `intercept` is in `specification` but not in `dataframe` and
    adds the required column to dataframe. Note this function is not
    idempotent--it alters the original argument, `dataframe`.

    Parameters
    ----------
    specification : an iterable that has a `__contains__` method.
    dataframe : pandas DataFrame.
        Dataframe containing the data for the choice model to be estimated.

    Returns
    -------
    None.
    """
    if "intercept" in specification and "intercept" not in dataframe.columns:
        dataframe["intercept"] = 1.0

    return None


def check_num_rows_of_parameter_array(param_array, correct_num_rows, title):
    """
    Ensures that `param_array.shape[0]` has the correct magnitude. Raises a
    helpful ValueError if otherwise.

    Parameters
    ----------
    param_array : ndarray.
    correct_num_rows : int.
        The int that `param_array.shape[0]` should equal.
    title : str.
        The 'name' of the param_array whose shape is being checked.

    Results
    -------
    None.
    """
    if param_array.shape[0] != correct_num_rows:
        msg = "{}.shape[0] should equal {}, but it does not"
        raise ValueError(msg.format(title, correct_num_rows))

    return None


def check_type_and_size_of_param_list(param_list, expected_length):
    """
    Ensure that param_list is a list with the expected length. Raises a helpful
    ValueError if this is not the case.
    """
    try:
        assert isinstance(param_list, list)
        assert len(param_list) == expected_length
    except AssertionError:
        msg = "param_list must be a list containing {} elements."
        raise ValueError(msg.format(expected_length))

    return None


def check_type_of_param_list_elements(param_list):
    """
    Ensures that all elements of param_list are ndarrays or None. Raises a
    helpful ValueError if otherwise.
    """
    try:
        assert isinstance(param_list[0], np.ndarray)
        assert all([(x is None or isinstance(x, np.ndarray))
                    for x in param_list])
    except AssertionError:
        msg = "param_list[0] must be a numpy array."
        msg_2 = "All other elements must be numpy arrays or None."
        total_msg = msg + "\n" + msg_2
        raise TypeError(total_msg)

    return None


def check_num_columns_in_param_list_arrays(param_list):
    """
    Ensure that each array in param_list, that is not None, has the same number
    of columns. Raises a helpful ValueError if otherwise.

    Parameters
    ----------
    param_list : list of ndarrays or None.

    Returns
    -------
    None.
    """
    try:
        num_columns = param_list[0].shape[1]
        assert all([x is None or (x.shape[1] == num_columns)
                    for x in param_list])
    except AssertionError:
        msg = "param_list arrays should have equal number of columns."
        raise ValueError(msg)

    return None


def check_dimensional_equality_of_param_list_arrays(param_list):
    """
    Ensures that all arrays in param_list have the same dimension, and that
    this dimension is either 1 or 2 (i.e. all arrays are 1D arrays or all
    arrays are 2D arrays.) Raises a helpful ValueError if otherwise.

    Parameters
    ----------
    param_list : list of ndarrays or None.

    Returns
    -------
    None.
    """
    try:
        num_dimensions = len(param_list[0].shape)
        assert num_dimensions in [1, 2]
        assert all([(x is None or (len(x.shape) == num_dimensions))
                    for x in param_list])
    except AssertionError:
        msg = "Each array in param_list should be 1D or 2D."
        msg_2 = "And all arrays should have the same number of dimensions."
        total_msg = msg + "\n" + msg_2
        raise ValueError(total_msg)

    return None


def check_for_choice_col_based_on_return_long_probs(return_long_probs,
                                                    choice_col):
    """
    Ensure that if return_long_probs is False then choice_col is not None.
    Raise a helpful ValueError if otherwise.

    Parameters
    ----------
    return_long_probs : bool.
            Indicates whether or not the long format probabilities (a 1D numpy
            array with one element per observation per available alternative)
            should be returned.
    choice_col : str or None.
        Denotes the column in `data` which contains a one if the
        alternative pertaining to the given row was the observed outcome
        for the observation pertaining to the given row and a zero
        otherwise.

    Returns
    -------
    None.
    """
    if not return_long_probs and choice_col is None:
        msg = "If return_long_probs == False, then choice_col cannote be None."
        raise ValueError(msg)
    else:
        return None


def ensure_all_mixing_vars_are_in_the_name_dict(mixing_vars,
                                                name_dict,
                                                ind_var_names):
    """
    Ensures that all of the variables listed in `mixing_vars` are present in
    `ind_var_names`. Raises a helpful ValueError if otherwise.

    Parameters
    ----------
    mixing_vars : list of strings, or None.
        Each string denotes a parameter to be treated as a random variable.
    name_dict : OrderedDict or None.
        Contains the specification relating column headers in one's data (i.e.
        the keys of the OrderedDict) to the index coefficients to be estimated
        based on this data (i.e. the values of each key).
    ind_var_names : list of strings.
        Each string denotes an index coefficient (i.e. a beta) to be estimated.

    Returns
    -------
    None.
    """
    if mixing_vars is None:
        return None

    # Determine the strings in mixing_vars that are missing from ind_var_names
    problem_names = [variable_name for variable_name in mixing_vars
                     if variable_name not in ind_var_names]

    # Create error messages for the case where we have a name dictionary and
    # the case where we do not have a name dictionary.
    msg_0 = "The following parameter names were not in the values of the "
    msg_1 = "passed name dictionary: \n{}"
    msg_with_name_dict = msg_0 + msg_1.format(problem_names)

    msg_2 = "The following paramter names did not match any of the default "
    msg_3 = "names generated for the parameters to be estimated: \n{}"
    msg_4 = "The default names that were generated were: \n{}"
    msg_without_name_dict = (msg_2 +
                             msg_3.format(problem_names) +
                             msg_4.format(ind_var_names))

    # Raise a helpful ValueError if any mixing_vars were missing from
    # ind_var_names
    if problem_names != []:
        if name_dict:
            raise ValueError(msg_with_name_dict)
        else:
            raise ValueError(msg_without_name_dict)

    return None


def ensure_all_alternatives_are_chosen(alt_id_col, choice_col, dataframe):
    """
    Ensures that all of the available alternatives in the dataset are chosen at
    least once (for model identification). Raises a ValueError otherwise.

    Parameters
    ----------
    alt_id_col : str.
        Should denote the column in `dataframe` that contains the alternative
        identifiers for each row.
    choice_col : str.
        Should denote the column in `dataframe` that contains the ones and
        zeros that denote whether or not the given row corresponds to the
        chosen alternative for the given individual.
    dataframe : pandas dataframe.
        Should contain the data being used to estimate the model, as well as
        the headers denoted by `alt_id_col` and `choice_col`.

    Returns
    -------
    None.
    """
    all_ids = set(dataframe[alt_id_col].unique())
    chosen_ids = set(dataframe.loc[dataframe[choice_col] == 1,
                                   alt_id_col].unique())
    non_chosen_ids = all_ids.difference(chosen_ids)
    if len(non_chosen_ids) != 0:
        msg = ("The following alternative ID's were not chosen in any choice "
               "situation: \n{}")
        raise ValueError(msg.format(non_chosen_ids))

    return None


def compute_aic(model_object):
    """
    Compute the Akaike Information Criteria for an estimated model.

    Parameters
    ----------
    model_object : an MNDC_Model (multinomial discrete choice model) instance.
        The model should have already been estimated.
        `model_object.log_likelihood` should be a number, and
        `model_object.params` should be a pandas Series.

    Returns
    -------
    aic : float.
        The AIC for the estimated model.

    Notes
    -----
    aic = -2 * log_likelihood + 2 * num_estimated_parameters

    References
    ----------
    Akaike, H. (1974). 'A new look at the statistical identification model',
        IEEE Transactions on Automatic Control 19, 6: 716-723.
    """
    assert isinstance(model_object.params, pd.Series)
    assert isinstance(model_object.log_likelihood, Number)

    return -2 * model_object.log_likelihood + 2 * model_object.params.size


def compute_bic(model_object):
    """
    Compute the Bayesian Information Criteria for an estimated model.

    Parameters
    ----------
    model_object : an MNDC_Model (multinomial discrete choice model) instance.
        The model should have already been estimated.
        `model_object.log_likelihood` and `model_object.nobs` should be a
        number, and `model_object.params` should be a pandas Series.

    Returns
    -------
    bic : float.
        The BIC for the estimated model.

    Notes
    -----
    bic = -2 * log_likelihood + log(num_observations) * num_parameters

    The original BIC was introduced as (-1 / 2) times the formula above.
    However, for model comparison purposes, it does not matter if the
    goodness-of-fit measure is multiplied by a constant across all models being
    compared. Moreover, the formula used above allows for a common scale
    between measures such as the AIC, BIC, DIC, etc.

    References
    ----------
    Schwarz, G. (1978), 'Estimating the dimension of a model', The Annals of
        Statistics 6, 2: 461–464.
    """
    assert isinstance(model_object.params, pd.Series)
    assert isinstance(model_object.log_likelihood, Number)
    assert isinstance(model_object.nobs, Number)

    log_likelihood = model_object.log_likelihood
    num_obs = model_object.nobs
    num_params = model_object.params.size

    return -2 * log_likelihood + np.log(num_obs) * num_params


# Create a basic class that sets the structure for the discrete outcome models
# to be specified later. MNDC stands for MultiNomial Discrete Choice.
class MNDC_Model(object):
    """
    Parameters
    ----------
    data : str or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col : str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    intercept_ref_pos : int, optional.
         Valid only when the intercepts being estimated are not part of the
         index. Specifies the alternative in the ordered array of unique
         alternative ids whose intercept or alternative-specific constant is
         not estimated, to ensure model identifiability. Default == None.
    shape_ref_pos : int, optional.
        Specifies the alternative in the ordered array of unique alternative
        ids whose shape parameter is not estimated, to ensure model
        identifiability. Default == None.
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.
    intercept_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternatives in data, minus 1. Each
        element of the list should be a string--the name of the corresponding
        alternative's intercept term, in sorted order of the possible
        alternative IDs. If None is passed, the resulting names that are shown
        in the estimation results will be
        `["Outside_ASC_{}".format(x) for x in shape_names]`. Default = None.
    shape_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternative IDs in data. Each element of
        the list should be a string denoting the name of the corresponding
        shape parameter for the given alternative, in sorted order of the
        possible alternative IDs. Default == None.
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    mixing_id_col : str, or None, optional.
        Should be a column heading in `data`. Should denote the column in
        `data` which contains the identifiers of the units of observation over
        which the coefficients of the model are thought to be randomly
        distributed. If `model_type == "Mixed Logit"`, then `mixing_id_col`
        must be passed. Default == None.
    mixing_vars : list, or None, optional.
        All elements of the list should be strings. Each string should be
        present in the values of `names.values()` and they're associated
        variables should only be index variables (i.e. part of the design
        matrix). If `model_type == "Mixed Logit"`, then `mixing_vars` must be
        passed. Default == None.
    model_type : str, optional.
        Denotes the model type of the choice model being instantiated.
        Default == "".
    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 intercept_ref_pos=None,
                 shape_ref_pos=None,
                 names=None,
                 intercept_names=None,
                 shape_names=None,
                 nest_spec=None,
                 mixing_vars=None,
                 mixing_id_col=None,
                 model_type=""):
        dataframe = get_dataframe_from_data(data)

        ##########
        # Make sure all necessary columns are in the dataframe
        ##########
        ensure_columns_are_in_dataframe([alt_id_col, obs_id_col, choice_col],
                                        dataframe,
                                        '[alt_id_col, obs_id_col, choice_col]',
                                        'data')

        ##########
        # Make sure the various 'name' arguments are of the correct lengths
        ##########
        # Get a sorted array of all possible alternative ids in the dataset
        all_ids = np.sort(dataframe[alt_id_col].unique())

        # For model identification, ensure that the number of chosen
        # alternatives equals the total number of alternatives available in the
        # dataset. Currently commented out because many tests need to be
        # rewritten to cope with this constraint being made explicit.
        # ensure_all_alternatives_are_chosen(alt_id_col, choice_col, dataframe)

        # Check for correct length of shape_names and intercept_names
        name_and_ref_args = [(shape_names, shape_ref_pos, "shape_names"),
                             (intercept_names,
                              intercept_ref_pos,
                              "intercept_names")]
        for alt_param_names, alt_ref_pos, param_string in name_and_ref_args:
            if alt_param_names is not None:
                ensure_ref_position_is_valid(alt_ref_pos,
                                             len(all_ids),
                                             param_string)

                alt_params_not_estimated = 0 if alt_ref_pos is None else 1

                length_args = [alt_param_names,
                               len(all_ids),
                               alt_params_not_estimated,
                               param_string]

                check_length_of_shape_or_intercept_names(*length_args)

        ##########
        # Check for validity of the nest_spec argument if necessary
        ##########
        if nest_spec is not None:
            ensure_object_is_ordered_dict(nest_spec, "nest_spec")

            check_type_of_nest_spec_keys_and_values(nest_spec)

            check_for_empty_nests_in_nest_spec(nest_spec)

            # Collect all lists of alternative ids belonging to each nest
            list_elements = reduce(lambda x, y: x + y,
                                   [nest_spec[key] for key in nest_spec])

            ensure_alt_ids_in_nest_spec_are_ints(nest_spec, list_elements)

            ensure_alt_ids_are_only_in_one_nest(nest_spec, list_elements)

            ensure_all_alt_ids_have_a_nest(nest_spec, list_elements, all_ids)

            ensure_nest_alts_are_valid_alts(nest_spec, list_elements, all_ids)

        ##########
        # Add an intercept column to the data if necessary based on the model
        # specification.
        ##########
        add_intercept_to_dataframe(specification, dataframe)

        ##########
        # Make sure all the columns in the specification dict keys are all
        # in the dataframe
        ##########
        ensure_specification_cols_are_in_dataframe(specification, dataframe)

        ##########
        # Make sure that the columns we are using in the specification are all
        # numeric and exclude positive or negative infinity variables.
        ##########
        ensure_valid_nums_in_specification_cols(specification, dataframe)

        ##########
        # Create the design matrix for this model
        ##########
        design_res = create_design_matrix(dataframe,
                                          specification,
                                          alt_id_col,
                                          names=names)

        ##########
        # Make sure that the passed mixing variables are valid.
        # Note that design_res[1] contains the index variable names.
        ##########
        ensure_all_mixing_vars_are_in_the_name_dict(mixing_vars,
                                                    names,
                                                    design_res[1])
        ##########
        # Store needed data
        ##########
        self.data = dataframe
        self.name_spec = names
        self.design = design_res[0]
        self.ind_var_names = design_res[1]
        self.alt_id_col = alt_id_col
        self.obs_id_col = obs_id_col
        self.choice_col = choice_col
        self.specification = specification
        self.alt_IDs = dataframe[alt_id_col].values
        self.choices = dataframe[choice_col].values
        self.model_type = model_type
        self.shape_names = shape_names
        self.intercept_names = intercept_names
        self.shape_ref_position = shape_ref_pos
        self.intercept_ref_position = intercept_ref_pos
        self.nest_names = (list(nest_spec.keys())
                           if nest_spec is not None else None)
        self.nest_spec = nest_spec
        self.mixing_id_col = mixing_id_col
        self.mixing_vars = mixing_vars
        if mixing_vars is not None:
            mixing_pos = convert_mixing_names_to_positions(mixing_vars,
                                                           self.ind_var_names)
        else:
            mixing_pos = None
        self.mixing_pos = mixing_pos

        return None

    def get_mappings_for_fit(self, dense=False):
        """
        Parameters
        ----------
        dense : bool, optional.
            Dictates if sparse matrices will be returned or dense numpy arrays.

        Returns
        -------
        mapping_dict : OrderedDict.
            Keys will be `["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
            "rows_to_nests"]`. The value for `rows_to_obs` will map the rows of
            the `long_form` to the unique observations (on the columns) in
            their order of appearance. The value for `rows_to_alts` will map
            the rows of the `long_form` to the unique alternatives which are
            possible in the dataset (on the columns), in sorted order--not
            order of appearance. The value for `chosen_row_to_obs`, if not
            None, will map the rows of the `long_form` that contain the chosen
            alternatives to the specific observations those rows are associated
            with (denoted by the columns). The value of `rows_to_nests`, if not
            None, will map the rows of the `long_form` to the nest (denoted by
            the column) that contains the row's alternative. If `dense==True`,
            the returned values will be dense numpy arrays. Otherwise, the
            returned values will be scipy sparse arrays.
        """
        return create_long_form_mappings(self.data,
                                         self.obs_id_col,
                                         self.alt_id_col,
                                         choice_col=self.choice_col,
                                         nest_spec=self.nest_spec,
                                         mix_id_col=self.mixing_id_col,
                                         dense=dense)

    def _store_basic_estimation_results(self, results_dict):
        """
        Extracts the basic estimation results (i.e. those that need no further
        calculation or logic applied to them) and stores them on the model
        object.

        Parameters
         ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
            `["final_log_likelihood", "chosen_probs", "long_probs",
              "residuals", "ind_chi_squareds", "sucess", "message",
              "rho_squared", "rho_bar_squared", "log_likelihood_null"]`

        Returns
        -------
        None.
        """
        # Store the log-likelilhood, fitted probabilities, residuals, and
        # individual chi-square statistics
        self.log_likelihood = results_dict["final_log_likelihood"]
        self.fitted_probs = results_dict["chosen_probs"]
        self.long_fitted_probs = results_dict["long_probs"]
        self.long_residuals = results_dict["residuals"]
        self.ind_chi_squareds = results_dict["ind_chi_squareds"]
        self.chi_square = self.ind_chi_squareds.sum()

        # Store the 'estimation success' of the optimization
        self.estimation_success = results_dict["success"]
        self.estimation_message = results_dict["message"]

        # Store the summary measures of the model fit
        self.rho_squared = results_dict["rho_squared"]
        self.rho_bar_squared = results_dict["rho_bar_squared"]

        # Store the initial and null log-likelihoods
        self.null_log_likelihood = results_dict["log_likelihood_null"]

        return None

    def _create_results_summary(self):
        """
        Create the dataframe that displays the estimation results, and store
        it on the model instance.

        Returns
        -------
        None.
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes = ["params",
                             "standard_errors",
                             "tvalues",
                             "pvalues",
                             "robust_std_errs",
                             "robust_t_stats",
                             "robust_p_vals"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes])
            assert all([isinstance(getattr(self, attr), pd.Series)
                        for attr in needed_attributes])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        self.summary = pd.concat((self.params,
                                  self.standard_errors,
                                  self.tvalues,
                                  self.pvalues,
                                  self.robust_std_errs,
                                  self.robust_t_stats,
                                  self.robust_p_vals), axis=1)

        return None

    def _record_values_for_fit_summary_and_statsmodels(self):
        """
        Store the various estimation results that are used to describe how well
        the estimated model fits the given dataset, and record the values that
        are needed for the statsmodels estimation results table. All values are
        stored on the model instance.

        Returns
        -------
        None.
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes = ["fitted_probs",
                             "params",
                             "log_likelihood",
                             "standard_errors"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        # Record the number of observations
        self.nobs = self.fitted_probs.shape[0]
        # This is the number of estimated parameters
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model
        # This is just the log-likelihood. The opaque name is used for
        # conformance with statsmodels
        self.llf = self.log_likelihood
        # This is just a repeat of the standard errors
        self.bse = self.standard_errors
        # These are the penalized measures of fit used for model comparison
        self.aic = compute_aic(self)
        self.bic = compute_bic(self)

        return None

    def _create_fit_summary(self):
        """
        Create and store a pandas series that will display to users the
        various statistics/values that indicate how well the estimated model
        fit the given dataset.

        Returns
        -------
        None.
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes = ["df_model",
                             "nobs",
                             "null_log_likelihood",
                             "log_likelihood",
                             "rho_squared",
                             "rho_bar_squared",
                             "estimation_message"]
        try:
            assert all([hasattr(self, attr) for attr in needed_attributes])
            assert all([getattr(self, attr) is not None
                        for attr in needed_attributes])
        except AssertionError:
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            raise NotImplementedError(msg + msg_2)

        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.null_log_likelihood,
                                      self.log_likelihood,
                                      self.rho_squared,
                                      self.rho_bar_squared,
                                      self.estimation_message],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood",
                                            "Rho-Squared",
                                            "Rho-Bar-Squared",
                                            "Estimation Message"])

        return None

    def _store_inferential_results(self,
                                   value_array,
                                   index_names,
                                   attribute_name,
                                   series_name=None,
                                   column_names=None):
        """
        Store the estimation results that relate to statistical inference, such
        as parameter estimates, standard errors, p-values, etc.

        Parameters
        ----------
        value_array : 1D or 2D ndarray.
            Contains the values that are to be stored on the model instance.
        index_names : list of strings.
            Contains the names that are to be displayed on the 'rows' for each
            value being stored. There should be one element for each value of
            `value_array.`
        series_name : string or None, optional.
            The name of the pandas series being created for `value_array.` This
            kwarg should be None when `value_array` is a 1D ndarray.
        attribute_name : string.
            The attribute name that will be exposed on the model instance and
            related to the passed `value_array.`
        column_names : list of strings, or None, optional.
            Same as `index_names` except that it pertains to the columns of a
            2D ndarray. When `value_array` is a 2D ndarray, There should be one
            element for each column of `value_array.` This kwarg should be None
            otherwise.

        Returns
        -------
        None. Stores a pandas series or dataframe on the model instance.
        """
        if len(value_array.shape) == 1:
            assert series_name is not None
            new_attribute_value = pd.Series(value_array,
                                            index=index_names,
                                            name=series_name)
        elif len(value_array.shape) == 2:
            assert column_names is not None
            new_attribute_value = pd.DataFrame(value_array,
                                               index=index_names,
                                               columns=column_names)

        setattr(self, attribute_name, new_attribute_value)

        return None

    def _store_generic_inference_results(self,
                                         results_dict,
                                         all_params,
                                         all_names):
        """
        Store the model inference values that are common to all choice models.
        This includes things like index coefficients, gradients, hessians,
        asymptotic covariance matrices, t-values, p-values, and robust versions
        of these values.

        Parameters
        ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
           `["utility_coefs", "final_gradient", "final_hessian",
             "fisher_info"]`.
            The "final_gradient", "final_hessian", and "fisher_info" values
            should be the gradient, hessian, and Fisher-Information Matrix of
            the log likelihood, evaluated at the final parameter vector.
        all_params : list of 1D ndarrays.
            Should contain the various types of parameters that were actually
            estimated.
        all_names : list of strings.
            Should contain names of each estimated parameter.

        Returns
        -------
        None. Stores all results on the model instance.
        """
        # Store the utility coefficients
        self._store_inferential_results(results_dict["utility_coefs"],
                                        index_names=self.ind_var_names,
                                        attribute_name="coefs",
                                        series_name="coefficients")

        # Store the gradient
        self._store_inferential_results(results_dict["final_gradient"],
                                        index_names=all_names,
                                        attribute_name="gradient",
                                        series_name="gradient")

        # Store the hessian
        self._store_inferential_results(results_dict["final_hessian"],
                                        index_names=all_names,
                                        attribute_name="hessian",
                                        column_names=all_names)

        # Store the variance-covariance matrix
        self._store_inferential_results(-1 * scipy.linalg.inv(self.hessian),
                                        index_names=all_names,
                                        attribute_name="cov",
                                        column_names=all_names)

        # Store ALL of the estimated parameters
        self._store_inferential_results(np.concatenate(all_params, axis=0),
                                        index_names=all_names,
                                        attribute_name="params",
                                        series_name="parameters")

        # Store the standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.cov)),
                                        index_names=all_names,
                                        attribute_name="standard_errors",
                                        series_name="std_err")

        # Store the t-stats of the estimated parameters
        self.tvalues = self.params / self.standard_errors
        self.tvalues.name = "t_stats"

        # Store the p-values
        p_vals = 2 * scipy.stats.norm.sf(np.abs(self.tvalues))
        self._store_inferential_results(p_vals,
                                        index_names=all_names,
                                        attribute_name="pvalues",
                                        series_name="p_values")

        # Store the fischer information matrix of estimated coefficients
        self._store_inferential_results(results_dict["fisher_info"],
                                        index_names=all_names,
                                        attribute_name="fisher_information",
                                        column_names=all_names)

        # Store the 'robust' variance-covariance matrix
        robust_covariance = calc_asymptotic_covariance(self.hessian,
                                                       self.fisher_information)
        self._store_inferential_results(robust_covariance,
                                        index_names=all_names,
                                        attribute_name="robust_cov",
                                        column_names=all_names)

        # Store the 'robust' standard errors
        self._store_inferential_results(np.sqrt(np.diag(self.robust_cov)),
                                        index_names=all_names,
                                        attribute_name="robust_std_errs",
                                        series_name="robust_std_err")

        # Store the 'robust' t-stats of the estimated coefficients
        self.robust_t_stats = self.params / self.robust_std_errs
        self.robust_t_stats.name = "robust_t_stats"

        # Store the 'robust' p-values
        one_sided_p_vals = scipy.stats.norm.sf(np.abs(self.robust_t_stats))
        self._store_inferential_results(2 * one_sided_p_vals,
                                        index_names=all_names,
                                        attribute_name="robust_p_vals",
                                        series_name="robust_p_values")

        return None

    def _store_optional_parameters(self,
                                   optional_params,
                                   name_list_attr,
                                   default_name_str,
                                   all_names,
                                   all_params,
                                   param_attr_name,
                                   series_name):
        """
        Extract the optional parameters from the `results_dict`, save them
        to the model object, and update the list of all parameters and all
        parameter names.

        Parameters
        ----------
        optional_params : 1D ndarray.
            The optional parameters whose values and names should be stored.
        name_list_attr : str.
            The attribute name on the model object where the names of the
            optional estimated parameters will be stored (if they exist).
        default_name_str : str.
            The name string that will be used to create generic names for the
            estimated parameters, in the event that the estimated parameters
            do not have names that were specified by the user. Should contain
            empty curly braces for use with python string formatting.
        all_names : list of strings.
            The current list of the names of the estimated parameters. The
            names of these optional parameters will be added to the beginning
            of this list.
        all_params : list of 1D ndarrays.
            Each array is a set of estimated parameters. The current optional
            parameters will be added to the beginning of this list.
        param_attr_name : str.
            The attribute name that will be used to store the optional
            parameter values on the model object.
        series_name : str.
            The string that will be used as the name of the series that
            contains the optional parameters.

        Returns
        -------
        (all_names, all_params) : tuple.
        """
        # Identify the number of optional parameters
        num_elements = optional_params.shape[0]

        # Get the names of the optional parameters
        parameter_names = getattr(self, name_list_attr)
        if parameter_names is None:
            parameter_names = [default_name_str.format(x) for x in
                               range(1, num_elements + 1)]

        # Store the names of the optional parameters in all_names
        all_names = list(parameter_names) + list(all_names)
        # Store the values of the optional parameters in all_params
        all_params.insert(0, optional_params)

        # Store the optional parameters on the model object
        self._store_inferential_results(optional_params,
                                        index_names=parameter_names,
                                        attribute_name=param_attr_name,
                                        series_name=series_name)
        return all_names, all_params

    def _adjust_inferential_results_for_parameter_constraints(self,
                                                              constraints):
        """
        Ensure that parameters that were constrained during estimation do not
        have any values showed for inferential results. After all, no inference
        was performed.

        Parameters
        ----------
        constraints : list of ints, or None.
            If list, should contain the positions in the array of all estimated
            parameters that were constrained to their initial values.

        Returns
        -------
        None.
        """
        if constraints is not None:
            # Ensure the model object has inferential results
            inferential_attributes = ["standard_errors",
                                      "tvalues",
                                      "pvalues",
                                      "robust_std_errs",
                                      "robust_t_stats",
                                      "robust_p_vals"]
            assert all([hasattr(self, x) for x in inferential_attributes])
            assert hasattr(self, "params")

            all_names = self.params.index.tolist()

            for series in [getattr(self, x) for x in inferential_attributes]:
                for pos in constraints:
                    series.loc[all_names[pos]] = np.nan

        return None

    def _check_result_dict_for_needed_keys(self, results_dict):
        """
        Ensure that `results_dict` has the needed keys to store all the
        estimation results. Raise a helpful ValueError otherwise.
        """
        missing_cols = [x for x in needed_result_keys if x not in results_dict]
        if missing_cols != []:
            msg = "The following keys are missing from results_dict\n{}"
            raise ValueError(msg.format(missing_cols))
        return None

    def _add_mixing_variable_names_to_individual_vars(self):
        """
        Ensure that the model objects mixing variables are added to its list of
        individual variables.
        """
        assert isinstance(self.ind_var_names, list)
        # Note that if one estimates a mixed logit model, then the mixing
        # variables will be added to individual vars. And if one estimates
        # the model again (perhaps from different starting values), then
        # an error will be raised when creating the coefs series because we
        # will have added the mixing variables twice. The condition below
        # should prevent this error.
        already_included = any(["Sigma " in x for x in self.ind_var_names])

        if self.mixing_vars is not None and not already_included:
            new_ind_var_names = ["Sigma " + x for x in self.mixing_vars]
            self.ind_var_names += new_ind_var_names
        return None

    def store_fit_results(self, results_dict):
        """
        Parameters
        ----------
        results_dict : dict.
            The estimation result dictionary that is output from
            scipy.optimize.minimize. In addition to the standard keys which are
            included, it should also contain the following keys:
           `["final_gradient", "final_hessian", "fisher_info",
             "final_log_likelihood", "chosen_probs", "long_probs", "residuals",
             "ind_chi_squareds"]`.
            The "final_gradient", "final_hessian", and "fisher_info" values
            should be the gradient, hessian, and Fisher-Information Matrix of
            the log likelihood, evaluated at the final parameter vector.

        Returns
        -------
        None. Will calculate and store a variety of estimation results and
        inferential statistics as attributes of the model instance.
        """
        # Check to make sure the results_dict has all the needed keys.
        self._check_result_dict_for_needed_keys(results_dict)

        # Store the basic estimation results that simply need to be transferred
        # from the results_dict to the model instance.
        self._store_basic_estimation_results(results_dict)

        # Account for attributes from the mixed logit model.
        if not hasattr(self, "design_3d"):
            self.design_3d = None

        # Initialize the lists of all parameter names and all parameter values
        # Note we add the new mixing variables to the list of index
        # coefficients after estimation so that we can correctly create the
        # design matrix during the estimation proces. The create_design_3d
        # function relies on the original list of independent variable names.
        self._add_mixing_variable_names_to_individual_vars()
        all_names = deepcopy(self.ind_var_names)
        all_params = [deepcopy(results_dict["utility_coefs"])]

        ##########
        # Figure out whether this model had nest, shape, or intercept
        # parameters and store each of these appropriately
        ##########
        if results_dict["intercept_params"] is not None:
            storage_args = [results_dict["intercept_params"],
                            "intercept_names",
                            "Outside_ASC_{}",
                            all_names,
                            all_params,
                            "intercepts",
                            "intercept_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.intercepts = None

        if results_dict["shape_params"] is not None:
            storage_args = [results_dict["shape_params"],
                            "shape_names",
                            "Shape_{}",
                            all_names,
                            all_params,
                            "shapes",
                            "shape_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.shapes = None

        if results_dict["nest_params"] is not None:
            storage_args = [results_dict["nest_params"],
                            "nest_names",
                            "Nest_Param_{}",
                            all_names,
                            all_params,
                            "nests",
                            "nest_parameters"]
            storage_results = self._store_optional_parameters(*storage_args)
            all_names, all_params = storage_results
        else:
            self.nests = None

        # Store the model results and values needed for model inference
        self._store_generic_inference_results(results_dict,
                                              all_params,
                                              all_names)

        # Adjust the inferential results to account for parameters that were
        # not actually estimated, i.e. parameters that were constrained.
        constraints = results_dict["constrained_pos"]
        self._adjust_inferential_results_for_parameter_constraints(constraints)

        # Store a summary dataframe of the estimation results
        self._create_results_summary()

        # Record values for the fit_summary and statsmodels table
        self._record_values_for_fit_summary_and_statsmodels()

        # Store a "Fit Summary"
        self._create_fit_summary()

        return None

    # Note that the function below is a placeholder and template for the
    # function to be placed in each model class.
    def fit_mle(self,
                init_vals,
                print_res=True,
                method="BFGS",
                loss_tol=1e-06,
                gradient_tol=1e-06,
                maxiter=1000,
                PMLE=None,
                PMLE_lambda=0,
                *args):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimizatin process with. There
            should be one value for each utility coefficient, outside intercept
            parameter, shape parameter, and nest parameter being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string which can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm
            which is used for this problem.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next which is needed to determine
            convergence. Default == 1e-06.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
            Default == 1e-06.
        PMLE: None or string value: ['LASSO', 'RIDGE' or 'Tikhonov']
            It determines if a Penalized Maximum Likelihood Estimation should be
            executed. The value of the parameter determines the type of PMLE. The
            None value states that no PMLE is executed. Default = None.
        PMLE_lambda : int, float, long, or None, optional.
            Lambda parameter for LASSO or ridge regression. It should be an int,
            float or long and determines the penalty for the optimization.
            Default = 0.

        Returns
        -------
        None. Saves estimation results to the model instance.
        """
        msg = "This model class' fit_mle method has not been constructed."
        raise NotImplementedError(msg)

    def print_summaries(self):
        """
        Returns None. Will print the measures of fit and the estimation results
        for the  model.
        """
        if hasattr(self, "fit_summary") and hasattr(self, "summary"):
            print("\n")
            print(self.fit_summary)
            print("=" * 30)
            print(self.summary)

        else:
            msg = "This {} object has not yet been estimated so there "
            msg_2 = "are no estimation summaries to print."
            raise NotImplementedError(msg.format(self.model_type) + msg_2)

        return None

    def conf_int(self, alpha=0.05, coefs=None, return_df=False):
        """
        Creates the dataframe or array of lower and upper bounds for the
        (1-alpha)% confidence interval of the estimated parameters. Used when
        creating the statsmodels summary.

        Parameters
        ----------
        alpha : float, optional.
            Should be between 0.0 and 1.0. Determines the (1-alpha)% confidence
            interval that will be reported. Default == 0.05.
        coefs : array-like, optional.
            Should contain strings that denote the coefficient names that one
            wants the confidence intervals for. Default == None because that
            will return the confidence interval for all variables.
        return_df : bool, optional.
            Determines whether the returned value will be a dataframe or a
            numpy array. Default = False.

        Returns
        -------
        pandas dataframe or ndarray.
            Depends on return_df kwarg. The first column contains the lower
            bound to the confidence interval whereas the second column contains
            the upper values of the confidence intervals.
        """

        # Get the critical z-value for alpha / 2
        z_critical = scipy.stats.norm.ppf(1.0 - alpha / 2.0,
                                          loc=0, scale=1)

        # Calculate the lower and upper values for the confidence interval.
        lower = self.params - z_critical * self.standard_errors
        upper = self.params + z_critical * self.standard_errors
        # Name the series of lower / upper values for the confidence interval.
        lower.name = "lower"
        upper.name = "upper"

        # Combine the various series.
        combined = pd.concat((lower, upper), axis=1)

        # Subset the combined dataframe if need be.
        if coefs is not None:
            combined = combined.loc[coefs, :]

        # Return the desired object, whether dataframe or array
        if return_df:
            return combined
        else:
            return combined.values

    def get_statsmodels_summary(self,
                                title=None,
                                alpha=.05):
        """
        Parameters
        ----------
        title : str, or None, optional.
            Will be the title of the returned summary. If None, the default
            title is used.
        alpha : float, optional.
            Should be between 0.0 and 1.0. Determines the width of the
            displayed, (1 - alpha)% confidence interval.

        Returns
        -------
        statsmodels.summary object or None.
        """
        try:
            # Get the statsmodels Summary class
            from statsmodels.iolib.summary import Summary
        except ImportError:
            print("statsmodels not installed. Resorting to standard summary")
            return self.print_summaries()

        if not hasattr(self, "estimation_success"):
            msg = "Must estimate a model before a summary can be returned."
            raise NotImplementedError(msg)

        # Get an instantiation of the Summary class.
        smry = Summary()

        # Get the yname and yname_list.
        # Note I'm not really sure what the yname_list is.
        new_yname, new_yname_list = self.choice_col, None

        # Get the model name
        model_name = self.model_type

        ##########
        # Note the following commands are basically directly from
        # statsmodels.discrete.discrete_model
        ##########
        top_left = [('Dep. Variable:', None),
                    ('Model:', [model_name]),
                    ('Method:', ['MLE']),
                    ('Date:', None),
                    ('Time:', None),
                    ('AIC:', ["{:,.3f}".format(self.aic)]),
                    ('BIC:', ["{:,.3f}".format(self.bic)])
                    ]

        top_right = [('No. Observations:', ["{:,}".format(self.nobs)]),
                     ('Df Residuals:', ["{:,}".format(self.df_resid)]),
                     ('Df Model:', ["{:,}".format(self.df_model)]),
                     ('Pseudo R-squ.:',
                      ["{:.3f}".format(self.rho_squared)]),
                     ('Pseudo R-bar-squ.:',
                      ["{:.3f}".format(self.rho_bar_squared)]),
                     ('Log-Likelihood:', ["{:,.3f}".format(self.llf)]),
                     ('LL-Null:',
                      ["{:,.3f}".format(self.null_log_likelihood)]),
                     ]

        if title is None:
            title = model_name + ' ' + "Regression Results"

        xnames = self.params.index.tolist()

        # for top of table
        smry.add_table_2cols(self,
                             gleft=top_left,
                             gright=top_right,  # [],
                             yname=new_yname,
                             xname=xnames,
                             title=title)
        # for parameters, etc
        smry.add_table_params(self,
                              yname=[new_yname_list],
                              xname=xnames,
                              alpha=alpha,
                              use_t=False)
        return smry

    def check_param_list_validity(self, param_list):
        """
        Parameters
        ----------
        param_list : list.
            Contains four elements, each being a numpy array. Either all of the
            arrays should be 1D or all of the arrays should be 2D. If 2D, the
            arrays should have the same number of columns. Each column being a
            particular set of parameter values that one wants to predict with.
            The first element in the list should be the index coefficients. The
            second element should contain the 'outside' intercept parameters if
            there are any, or None otherwise. The third element should contain
            the shape parameters if there are any or None otherwise. The fourth
            element should contain the nest coefficients if there are any or
            None otherwise. Default == None.

        Returns
        -------
        None. Will check whether `param_list` and its elements meet all
        requirements specified above and required for correct calculation of
        the probabilities to be predicted.
        """
        if param_list is None:
            return None

        # Make sure there are four elements in param_list
        check_type_and_size_of_param_list(param_list, 4)

        # Make sure each element in the list is a numpy array or is None
        check_type_of_param_list_elements(param_list)

        # Make sure each array in param_list has the same number of dimensions
        check_dimensional_equality_of_param_list_arrays(param_list)

        # If using 2D arrays, ensure each array has the same number of columns.
        if len(param_list[0].shape) == 2:
            check_num_columns_in_param_list_arrays(param_list)

        # Make sure each array has the correct number of elements
        num_index_coefs = len(self.ind_var_names)

        check_num_rows_of_parameter_array(param_list[0],
                                          num_index_coefs,
                                          'param_list[0]')

        if param_list[1] is not None:
            num_intercepts = (0 if self.intercept_names is None else
                              len(self.intercept_names))

            check_num_rows_of_parameter_array(param_list[1],
                                              num_intercepts,
                                              'param_list[1]')

        if param_list[2] is not None:
            num_shapes = (0 if self.shape_names is None else
                          len(self.shape_names))

            check_num_rows_of_parameter_array(param_list[2],
                                              num_shapes,
                                              'param_list[2]')

        if param_list[3] is not None:
            num_nests = (0 if self.nest_names is None else
                         len(self.nest_names))

            check_num_rows_of_parameter_array(param_list[3],
                                              num_nests,
                                              'param_list[3]')

        return None

    def predict(self,
                data,
                param_list=None,
                return_long_probs=True,
                choice_col=None,
                num_draws=None,
                seed=None):
        """
        Parameters
        ----------
        data : string or pandas dataframe.
            If string, data should be an absolute or relative path to a CSV
            file containing the long format data for this choice model. Note
            long format is has one row per available alternative for each
            observation. If pandas dataframe, the dataframe should be the long
            format data for the choice model. The data should include all of
            the same columns as the original data used to construct the choice
            model, with the sole exception of the "intercept" column. If needed
            the "intercept" column will be dynamically created.
        param_list : list, optional.
            Contains four elements, each being a numpy array or None. Either
            all of the arrays should be 1D or all of the arrays should be 2D.
            If 2D, the arrays should have the same number of columns. Each
            column should be a particular set of parameter values that one
            wants to predict with. The first element in the list should
            contain the index coefficients. The second element should contain
            the 'outside' intercept parameters if there are any, or None
            otherwise. The third element should contain the shape parameters
            if there are any or None otherwise. The fourth element should
            contain the nest coefficients if there are any or None otherwise.
            Default == None.
        return_long_probs : bool, optional.
            Indicates whether or not the long format probabilities (a 1D numpy
            array with one element per observation per available alternative)
            should be returned. Default == True.
        choice_col : str, optional.
            Denotes the column in `data` which contains a one if the
            alternative pertaining to the given row was the observed outcome
            for the observation pertaining to the given row and a zero
            otherwise. Default == None.
        num_draws : int, or None, optional.
            Should be greater than zero. Denotes the number of draws that we
            are making from each normal distribution. This kwarg is only used
            if self.model_type == "Mixed Logit Model". Default == None.
        seed : int, or None, optional.
            If an int is passed, it should be greater than zero. Denotes the
            value to be used in seeding the random generator used to generate
            the draws from the normal distribution. This kwarg is only used if
            self.model_type == "Mixed Logit Model". Default == None.

        Returns
        -------
        numpy array or tuple of two numpy arrays.
            If `choice_col` is passed AND `return_long_probs is True`, then the
            tuple `(chosen_probs, long_probs)` is returned. If
            `return_long_probs is True` and `chosen_row_to_obs is None`, then
            `long_probs` is returned. If `chosen_row_to_obs` is passed and
            `return_long_probs is False` then `chosen_probs` is returned.

            `chosen_probs` is a 1D numpy array of shape (num_observations,).
            Each element is the probability of the corresponding observation
            being associated with its realized outcome.

            `long_probs` is a 1D numpy array with one element per observation
            per available alternative for that observation. Each element is the
            probability of the corresponding observation being associated with
            that rows corresponding alternative.

            It is NOT valid to have `chosen_row_to_obs == None` and
            `return_long_probs == False`.
        """
        # Get the dataframe of observations we'll be predicting on
        dataframe = get_dataframe_from_data(data)

        # Determine the conditions under which we will add an intercept column
        # to our long format dataframe.
        add_intercept_to_dataframe(self.specification, dataframe)

        # Make sure the necessary columns are in the long format dataframe
        for column in [self.alt_id_col,
                       self.obs_id_col,
                       self.mixing_id_col]:
            if column is not None:
                ensure_columns_are_in_dataframe([column], dataframe)

        # If param_list is passed, check the validity of its elements
        self.check_param_list_validity(param_list)

        # Check validity of the return_long_probs and choice_col kwargs
        check_for_choice_col_based_on_return_long_probs(return_long_probs,
                                                        choice_col)

        # Get the new column of alternative IDs and get the new design matrix
        new_alt_IDs = dataframe[self.alt_id_col].values

        new_design_res = create_design_matrix(dataframe,
                                              self.specification,
                                              self.alt_id_col,
                                              names=self.name_spec)

        new_design = new_design_res[0]

        # Get the new mappings between the alternatives and observations
        mapping_res = create_long_form_mappings(dataframe,
                                                self.obs_id_col,
                                                self.alt_id_col,
                                                choice_col=choice_col,
                                                nest_spec=self.nest_spec,
                                                mix_id_col=self.mixing_id_col)

        new_rows_to_obs = mapping_res["rows_to_obs"]
        new_rows_to_alts = mapping_res["rows_to_alts"]
        new_chosen_to_obs = mapping_res["chosen_row_to_obs"]
        new_rows_to_nests = mapping_res["rows_to_nests"]
        new_rows_to_mixers = mapping_res["rows_to_mixers"]

        # Get the parameter arrays to be used in calculating the probabilities
        if param_list is None:
            new_index_coefs = self.coefs.values
            new_intercepts = (self.intercepts.values if self.intercepts
                              is not None else None)
            new_shape_params = (self.shapes.values if self.shapes
                                is not None else None)
            new_nest_coefs = (self.nests.values if self.nests
                              is not None else None)
        else:
            new_index_coefs = param_list[0]
            new_intercepts = param_list[1]
            new_shape_params = param_list[2]
            new_nest_coefs = param_list[3]

        # Get the probability of each observation choosing each available
        # alternative
        if self.model_type == "Nested Logit Model":
            # Get the 'natural' nest coefficients for prediction
            new_natural_nests = naturalize_nest_coefs(new_nest_coefs)
            # Determine the return string for the nested logit model
            if return_long_probs:
                return_string = "long_probs"
            else:
                return_string = "chosen_probs"
            # This condition accounts for the fact that we have a different
            # functional interface for nested vs non-nested models
            return calc_nested_probs(new_natural_nests,
                                     new_index_coefs,
                                     new_design,
                                     new_rows_to_obs,
                                     new_rows_to_nests,
                                     chosen_row_to_obs=new_chosen_to_obs,
                                     return_type=return_string)

        elif self.model_type == "Mixed Logit Model":
            ##########
            # This condition accounts for the fact that Mixed Logit models have
            # a different functional interface than the standard logit-type
            # models.
            ##########
            # Get the draws for each random coefficient
            num_mixing_units = new_rows_to_mixers.shape[1]
            draw_list = mlc.get_normal_draws(num_mixing_units,
                                             num_draws,
                                             len(self.mixing_pos),
                                             seed=seed)
            # Calculate the 3D design matrix for the prediction.
            design_args = (new_design,
                           draw_list,
                           self.mixing_pos,
                           new_rows_to_mixers)
            new_design_3d = mlc.create_expanded_design_for_mixing(*design_args)
            # Calculate the desired probabilities for the mixed logit model.
            prob_args = (new_index_coefs,
                         new_design_3d,
                         new_alt_IDs,
                         new_rows_to_obs,
                         new_rows_to_alts,
                         self.utility_transform)
            prob_kwargs = {"intercept_params": new_intercepts,
                           "shape_params": new_shape_params,
                           "chosen_row_to_obs": new_chosen_to_obs,
                           "return_long_probs": return_long_probs}
            prob_array = calc_probabilities(*prob_args, **prob_kwargs)
            return prob_array.mean(axis=1)

        else:
            return calc_probabilities(new_index_coefs,
                                      new_design,
                                      new_alt_IDs,
                                      new_rows_to_obs,
                                      new_rows_to_alts,
                                      self.utility_transform,
                                      intercept_params=new_intercepts,
                                      shape_params=new_shape_params,
                                      chosen_row_to_obs=new_chosen_to_obs,
                                      return_long_probs=return_long_probs)


    def _dataframe_with_predictions(self, data):
        """
        For a given input long dataframe containing a set of observations, it
        returns the same dataframe with two new columns, one with the predicted
        probability for each alternative for each decision-maker, and another
        with the predicted alternative for a given decision-maker (i.e. the one
        with the highest associated probability).

        Parameters
        ----------
        data : pandas dataframe.
            The dataset in long format for which the probabilities of choosing
            each alternative will be computed.

        Returns
        -------
        predictions : pandas dataframe.
            The input `data` datafame with two new columns:
                - probability : It is a value between 0 and 1. It denotes the
                    probability that the decison-maker chooses that alternative.
                    The sum of the probabilities for all the alternatives for any
                    decision-maker must be 1.
                - predicted : An integer value 0 or 1. It takes the value 1 for
                    the alternative which is predicted to be selected by a
                    decion-maker.
        """
        # Check the validity of the parameters
        ensure_columns_are_in_dataframe([self.obs_id_col,
                                         self.alt_id_col,
                                         self.choice_col],
                                        data,
                                        data_title='data')

        predictions = data[[self.obs_id_col,
                            self.alt_id_col,
                            self.choice_col]].copy()
        predictions['probability'] = self.predict(data)

        # Get the index for the predited alternative for each observation
        idx_higher_prob = predictions.groupby(self.obs_id_col)['probability'].\
            transform(max) == predictions['probability']

        # Append a new column to the dataframe which contains 1 if that
        # alternative was predected by the model or 0 in other case
        predictions['predicted'] = 0
        predictions.loc[idx_higher_prob, 'predicted'] = 1

        # Get all the observations where more than 1 alternative can be selected
        # as all have the same probability
        sum_selected = predictions.groupby(self.obs_id_col,as_index=False)['predicted'].sum()
        mult_choice = sum_selected.loc[sum_selected['predicted'] > 1, self.obs_id_col]

        # For each of the observations with more than 1 selected alternative,
        # select one of them randomly.
        for idx in mult_choice:
            sel_alts = list(predictions.loc[(predictions[self.obs_id_col] == idx) & \
                (predictions['predicted'] == 1), self.alt_id_col])
            selected = random.randint(0, len(sel_alts)-1)

            predictions.loc[(predictions[self.obs_id_col] == idx) & \
                (predictions[self.alt_id_col].isin(sel_alts)), 'predicted'] = 0

            predictions.loc[(predictions[self.obs_id_col] == idx) & \
                (predictions[self.alt_id_col] == sel_alts[selected]), 'predicted'] = 1

        return predictions


    def precision_recall_fscore(self,
                                data,
                                averaging=None,
                                beta=1):
        """
        Returns the precision, recall and F-score metrics used to evaluate a
        model.

        Parameters
        ----------
        data : pandas dataframe.
            The dataset in long format used to evaluate the model.
        averaging : str or None, optional.
            The type of averaging performed when computing the metrics.
            If different to None, then a averaging is applied when computing the
            precision, recall and F-score. It is mandatory for models with
            more than two alternatives (multi-class classification). The types
            of averaging performed on the data can be:
                - micro : Calculate metrics globally by counting the total true
                    positives, false negatives and false positives.
                - macro : Calculate metrics for each label, and find their
                    unweighted mean. This does not take label imbalance into
                    account.
            Default == None
        beta : float, optional.
            Weight of precision in harmonic mean.

        Returns
        -------
        tuple.
            Tuple composed of three element:
                - precision : float
                    The precision is the ratio ``tp / (tp + fp)`` where ``tp``
                    is the number of true positives and ``fp`` the number of
                    false positives. The precision is intuitively the ability of
                    the classifier not to label as positive a sample that is
                    negative.
                - recall : float
                    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is
                    the number of true positives and ``fn`` the number of false
                    negatives. The recall is intuitively the ability of the
                    classifier to find all the positive samples.
                - fscore : float
                    The F-beta score is the weighted harmonic mean of precision
                    and recall, reaching its optimal value at 1 and its worst
                    value at 0.
        """
        # Check the validity of the parameters
        ensure_columns_are_in_dataframe([self.obs_id_col,
                                         self.alt_id_col,
                                         self.choice_col],
                                        data,
                                        data_title='data')

        if beta <= 0:
            raise ValueError("beta parameter should be >0 in the F-beta score")

        precision = None
        recall = None
        fscore = None

        # List of different alternatives in data
        alternatives = list(data[self.alt_id_col].unique())
        # Total number of alternatives
        n_alternatives = len(alternatives)

        if n_alternatives == 1:
            msg = "Error! Only one alternative is specified in the dataset."
            raise ValueError(total_msg)

        if n_alternatives == 2 and averaging is not None:
            msg = "There is only two alternatives in the input dataset."
            msg_2 = "Therefore, averaging variable should not be specified."
            total_msg = " ".join([msg_1, msg_2])
            warnings.warn(total_msg)

        if (n_alternatives > 2 and averaging is None) \
            or (n_alternatives > 2 and averaging not in ['micro', 'macro']):
            msg_1 = "Error! This is a multiclass model with " +\
                "{} different classes.".format(n_alternatives)
            msg_2 = "averaging parameter should be provided."
            msg_3 = "Valid values are: [micro, marco]."
            total_msg = "\n".join([msg_1, msg_2, msg_3])
            raise ValueError(total_msg)

        # Get a dataframe with the predictions for each observation
        predictions = self._dataframe_with_predictions(data)

        # Create a contingency table for each alternative.
        # Contingency_tables is a list containing for each alternative a tuple
        # (tp, fp, fn, tn).
        contingency_tables = []
        for alt in alternatives:
            # Calculate the true positives for the alternative
            tp_alt = predictions[(predictions[self.alt_id_col] == alt) &\
                (predictions['predicted'] == 1)][self.choice_col].sum()
            # Calculate the false positives for the alternative
            fp_alt = predictions[(predictions[self.alt_id_col] == alt) &\
                (predictions['predicted'] == 1)][self.choice_col].count()-tp_alt
            # Calculate the false negatives for the alternative
            fn_alt = predictions[(predictions[self.alt_id_col] == alt) &\
                (predictions['predicted'] == 0)][self.choice_col].sum()
            # Calculate the true negatives for the alternative
            tn_alt = predictions[(predictions[self.alt_id_col] == alt) &\
                (predictions['predicted'] == 0)][self.choice_col].count()-fn_alt
            # Append tp, fp, fn and tn to contingency_tables
            contingency_tables.append((tp_alt, fp_alt, fn_alt, tn_alt))

        if n_alternatives == 2:
            elm = contingency_tables[0]
            precision = elm[0]/(elm[0]+elm[1])
            recall = elm[0]/(elm[0]+elm[2])
            fscore = ((1+beta**2)*elm[0]) / ((1+beta**2)*elm[0] \
                     + (beta**2)*elm[2] + elm[1])

        elif averaging == "micro":
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for elm in contingency_tables:
                tp += elm[0]
                fp += elm[1]
                fn += elm[2]
                tn += elm[3]
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            fscore = (1+beta**2)*precision*recall / ((beta**2)*precision+recall)

        elif averaging == "macro":
            precision_alt = 0
            recall_alt = 0
            for elm in contingency_tables:
                precision_alt += elm[0]/(elm[0]+ elm[1])# tp_alt/(tp_alt+fp_alt)
                recall_alt +=  elm[0]/( elm[0]+ elm[2]) # tp_alt/(tp_alt+fn_alt)
            precision = precision_alt/n_alternatives
            recall = recall_alt/n_alternatives
            fscore = (1+beta**2)*precision*recall / ((beta**2)*precision+recall)

        return (precision, recall, fscore)


    def confusion_matrix(self, data):
        """
        Compute confusion matrix to evaluate the accuracy of a model.

        By definition a confusion matrix C is such that C_{i, j} is equal to the
        number of observations known to be in group i but predicted to be in
        group j.

        Parameters
        ----------
        data : pandas dataframe.
            The dataset in long format used to evaluate the model.

        Returns
        -------
        confusion_matrix : pandas dataframe.
            Dataframe containing the confusion matrix. The columns represents
            the predicted alternatives, whereas the rows represents the true or
            actual alternatives.
        """
        # Check the validity of the parameters
        ensure_columns_are_in_dataframe([self.obs_id_col,
                                         self.alt_id_col,
                                         self.choice_col],
                                        data,
                                        data_title='data')

        # List of different alternatives in data
        alternatives = list(data[self.alt_id_col].unique())
        # Total number of alternatives
        n_alternatives = len(alternatives)

        confusion_matrix = pd.DataFrame(data=np.zeros((n_alternatives, n_alternatives)),
                                        columns=alternatives,
                                        index=alternatives,
                                        dtype=np.int)

        # Append to data the predicted alternatives per observation
        predictions = self._dataframe_with_predictions(data)

        # Iterate over all observations in data
        obs = list(predictions[self.obs_id_col].unique())
        for ob in obs:
            ob_prediction = predictions[predictions[self.obs_id_col] == ob]
            pred_alt = ob_prediction[ob_prediction['predicted'] == 1][self.alt_id_col]
            actual_alt = ob_prediction[ob_prediction[self.choice_col] == 1][self.alt_id_col]

            confusion_matrix.loc[actual_alt, pred_alt] += 1

        return confusion_matrix


    def get_utility(self, data):
        """
        Calculate the systematic utility for each alternative for each
        individual for a given dataframe.

        Parameters
        ----------
        data : pandas dataframe.
            It should be a long format data for the choice model. Note that
            long format has one row per available alternative for each
            observation. The data should include all of the same columns as the
            original data used to construct (fit) the choice model, with the
            sole exception of the "intercept" column. If needed the "intercept"
            column will be dynamically created.

        Returns
        -------
        numpy array.
            1D numpy array with one element per observation per available
            alternative for that observation. Each element is the calculated
            systematic utility for that alternative and individual.

            Given J alternatives, to get the systematic utility of alternative
            j and individual n, the (J*n + j) element of the array should be
            considered.
        """
        # Add an intercept column to the data if necessary based on the model
        # specification
        add_intercept_to_dataframe(self.specification, data)

        # Create a new design matrix based on the input data
        design_matrix = create_design_matrix(data, self.specification, \
            self.alt_id_col)[0]
        # Get the systematic utilities by multiplying the design matrix by the
        # model coefficients
        sys_utilities = design_matrix.dot(self.coefs.values)

        return sys_utilities


    def to_pickle(self, filepath):
        """
        Parameters
        ----------
        filepath : str.
            Should end in .pkl. If it does not, ".pkl" will be appended to the
            passed string.

        Returns
        -------
        None. Saves the model object to the location specified by `filepath`.
        """
        if not isinstance(filepath, str):
            raise ValueError("filepath must be a string.")

        if not filepath.endswith(".pkl"):
            filepath = filepath + ".pkl"

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print("Model saved to {}".format(filepath))

        return None
