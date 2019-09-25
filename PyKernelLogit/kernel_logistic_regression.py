# -*- coding: utf-8 -*-
"""

@module:    kernel_logistic_regression.py
@name:      Kernel Logistic Regression add-on
@author:    José Ángel Martín Baos
@summary:   Contains functions which allows to use non-linear parameter
            specifications based on kernel logistic regression.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import re
from scipy import optimize

from .choice_tools import *

from . import base_multinomial_cm_v2

from sklearn.gaussian_process import kernels


# Create a dictionary relating the kernel type parameter to the class from
# sklearn.gaussian_process.kernels that implements that kernel.
kernel_type_to_class = {"RBF": kernels.RBF,
                        "Matern": kernels.Matern,
                        "RationalQuadratic": kernels.RationalQuadratic,
                        "PairwiseKernel": kernels.PairwiseKernel,
                        "Product": kernels.Product,
                        "ExpSineSquared": kernels.ExpSineSquared,
                        "DotProduct": kernels.DotProduct,
                        "CompoundKernel": kernels.CompoundKernel,
                        "Sum": kernels.Sum,
                        "Exponentiation": kernels.Exponentiation}

valid_kernel_list = kernel_type_to_class.keys()


def ensure_valid_kernel_type(specified_kernel, kernel_type_list):
    """
    Checks to make sure that `specified_kernel` is in `kernel_type_list` and
    raises a helpful error if this is not the case.

    Parameters
    ----------
    specified_kernel : str.
        Denotes the user-specified kernel class that is to be checked.
    kernel_type_list : list of strings.
        Contains all of the valid kernel classes types.

    Returns
    -------
    None.
    """
    if specified_kernel not in kernel_type_list:
        msg_1 = "The kernel_type specified was not valid."
        msg_2 = "Valid kernel-types are {}".format(kernel_type_list)
        msg_3 = "The passed kernel_type was: {}".format(specified_kernel)
        total_msg = "\n".join([msg_1, msg_2, msg_3])
        raise ValueError(total_msg)
    return None


def ensure_valid_variables_passed_to_kernel_matrix(data,
                                                   variables,
                                                   alt_id_col,
                                                   data_title = 'X'):
    """
    Checks to make sure that the specified list of variables per alternative in
    `variables` is in `data` and has a correct format and raises a helpful error
    if this is not the case.

    Parameters
    ----------
    data : pandas dataframe.
        The dataset in long format passed to the kernel matrix. The dtypes of
        all columns should be numeric.
    variables : dict.
        Python dictionary where each key represents each of the alternatives
        IDs and its associated value contains a list of characteristics from
        the long dataframe `X` and `Z` to be used on that alternative.
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    data_title : str, optional.
        Denotes the title of the dataframe that is being checked to see whether
        it contains the passed columns. Default == 'X'

    Returns
    -------
    None.
    """
    if not isinstance(variables, dict):
        raise ValueError("variables must be a dictionary.")

    # Check if `variables` dict contains one key for each available alternative
    alternatives = data[alt_id_col].unique()
    for alternative in alternatives:
        if alternative not in variables.keys():
            msg = "Alternative {} is not specified in variables dictionary.".\
                format(alternative)
            raise ValueError(msg)

    for alt in variables.keys():
        # Obtain the list of variables to be considered for alternative i
        alt_variables = variables[alt]
        ensure_columns_are_in_dataframe(alt_variables,
                                        data,
                                        data_title=data_title)

    return None


def excluded_observation_from_alternative(data,
                                          alt_id_col,
                                          obs_id_col,
                                          alt):
    """
    Returns a list of observations that do not have alternative `alt` within
    their choice set.

    Parameters
    ----------
    data : pandas dataframe.
        Contains one row for each available alternative for each observation.
        Should have the specified `[obs_id_col, alt_id_col, choice_col]` column
        headings. The dtypes of all columns should be numeric.
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    obs_id_col : str.
        Denotes the column in `data` that contains the observation ID
        values for each row.
    alt : int.
        Indicates the alternative for which the excluded observations are
        computed.

    Returns
    -------
    excluded_obs : list.
        Will contain one element for each observation present in `data` but
        not appearing in the observations which involves alternative `alt`.
        All elements should be integers and corresponds with the value of the
        observation ID from the `obs_id_col` column.
    """
    # Get the total number of observation IDs
    max_obs = data[obs_id_col].nunique()
    # Get the list of different observation IDs in data
    list_obs = list(data[obs_id_col].unique())
    # Get the list of observation IDs in data for an specific alternative (alt)
    list_obs_alt = list(data[data[alt_id_col] == alt][obs_id_col])

    # Store which observation IDs from data do not contain the alternative alt
    excluded_obs = [x for x in list_obs if x not in list_obs_alt]

    return excluded_obs


def kernel_matrix(X,
                  alt_id_col,
                  obs_id_col,
                  variables,
                  kernel_type="RBF",
                  Z=None,
                  **kernel_kwargs):
    """
    Returns a list composed of one kernel matrix K_i per alternative i.

    Parameters
    ----------
    X : pandas dataframe.
        The dataset in long format for which the kernel matrix is computed,
        i.e. X_i.
        Contains one row for each available alternative for each observation.
        Should have the specified `[obs_id_col, alt_id_col, choice_col]` column
        headings. The dtypes of all columns should be numeric.
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    obs_id_col : str.
        Denotes the column in `data` that contains the observation ID
        values for each row.
    variables : dict.
        Python dictionary where each key represents each of the alternatives
        IDs and its associated value contains a list of characteristics from
        the long dataframe `X` to be used on that alternative.
    kernel_type : str, optional.
        The type of kernel function to be used. The kernel_matrix function
        relies on the scikit-learn Python package for the kernel functions.
        The value passed on this parameter specifies the scikit-learn kernel
        function which is going to be used to compute the kernel matrix.
        Default == "RBF".
    Z : pandas dataframe or None, optional.
        The kernel_matrix function allows to specify a reference matrix Z_i
        to be used when computing the kernel matrix. If no Z dataframe is
        specified, then the kernel matrix is computed taking as reference
        matrix Z_i the input matrix X_i. Default == None.

    Returns
    -------
    A list of one kernel matrix K_i for each alternative i.
    """
    # Check the validity of the parameters
    ensure_columns_are_in_dataframe([alt_id_col, obs_id_col],
                                    X,
                                    data_title='X')
    ensure_valid_variables_passed_to_kernel_matrix(X, variables, alt_id_col,
                                                   'X')
    ensure_valid_kernel_type(kernel_type, valid_kernel_list)

    if Z is not None:
        ensure_columns_are_in_dataframe([alt_id_col, obs_id_col],
                                        Z,
                                        data_title='Z')
        ensure_valid_variables_passed_to_kernel_matrix(Z, variables, alt_id_col,
                                                       'Z')

    # Get the list of different alternatives in X dataframe
    alternatives = X[alt_id_col].unique()
    # Get the total number of alternatives
    n_alternatives = X[alt_id_col].nunique()

    # Create a new kernel based on the kernel type selected
    kernel = kernel_type_to_class[kernel_type](**kernel_kwargs)

    # If no input dataframe Z is provided, Z will be the same as X
    if Z is None:
        Z = X

    # Initialize a list which contains the names for the Kernel Matrix columns
    columns = []
    for d in range(1, Z[obs_id_col].nunique() + 1):
        columns.append("K_i_%d" % d)

    # Initialize the list K that contains the kernel matrix per each alternative
    K = []

    # Obtain the Kernel Matrix for each alternative
    for alt in variables.keys():
        # Obtain the list of variables to be considered for alternative i
        alt_variables = variables[alt]

        # Obtain a submatrix X_alt and Z_alt from matrix X and Z, respectively,
        # with only the desired alternative `alt` and the selected variables
        X_alt = X[X[alt_id_col] == alt][alt_variables]
        Z_alt = Z[Z[alt_id_col] == alt][alt_variables]

        # Create the Kernel Matrix for alternative i
        K_aux = kernel(X_alt, Z_alt)

        # Obtain a list with the observations that do not contain alternative
        # `alt` inside their choice set in Z dataset
        excluded_obs_Z = excluded_observation_from_alternative(Z,
                                                               alt_id_col,
                                                               obs_id_col,
                                                               alt)

        # Fill with 0's the columns corresponding those observations for which
        # alternative `alt` is not in the choice set
        for exc_obs in excluded_obs_Z:
            K_aux = np.insert(K_aux, exc_obs - 1,
                              np.zeros(K_aux.shape[0]),
                              axis=1)

        # Convert K_aux from array to dataframe and insert the columns names
        K_aux = pd.DataFrame(data=K_aux, dtype=np.float64, columns=columns)

        # Add K_aux to the list of Kernels `K`
        K.append(K_aux)

    return K


def long_format_with_kernel_matrix(X,
                                   alt_id_col,
                                   obs_id_col,
                                   choice_col,
                                   variables,
                                   kernel_type="RBF",
                                   Z=None,
                                   **kernel_kwargs):
    """
    Returns a long format dataframe containing the kernel matrix for all the
    alternatives in the dataframe X.

    Parameters
    ----------
    X : pandas dataframe.
        The dataset in long format for which the kernel matrix is computed,
        i.e. X_i.
        Contains one row for each available alternative for each observation.
        Should have the specified `[obs_id_col, alt_id_col, choice_col]` column
        headings. The dtypes of all columns should be numeric.
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    obs_id_col : str.
        Denotes the column in `data` that contains the observation ID
        values for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    variables : dict.
        Python dictionary where each key represents each of the alternatives
        IDs and its associated value contains a list of characteristics from
        the long dataframe `X` to be used on that alternative.
    kernel_type : str, optional.
        The type of kernel function to be used. The kernel_matrix function
        relies on the scikit-learn Python package for the kernel functions.
        The value passed on this parameter specifies the scikit-learn kernel
        function which is going to be used to compute the kernel matrix.
        Default == "RBF".
    Z : pandas dataframe or None, optional.
        The kernel_matrix function allows to specify a reference matrix Z_i
        to be used when computing the kernel matrix. If no Z dataframe is
        specified, then the kernel matrix is computed taking as reference
        matrix Z_i the input matrix X_i. Default == None.

    Returns
    -------
    K_long_format : pandas dataframe.
        A long format dataframe that contains the kernel matrix for all the
        alternatives in the dataframe X.
    """
    # Check the validity of the parameters
    ensure_columns_are_in_dataframe([alt_id_col, obs_id_col, choice_col],
                                    X,
                                    data_title='X')
    ensure_valid_variables_passed_to_kernel_matrix(X, variables, alt_id_col,
                                                   'X')

    if Z is not None:
        ensure_columns_are_in_dataframe([alt_id_col, obs_id_col, choice_col],
                                        Z,
                                        data_title='Z')
        ensure_valid_variables_passed_to_kernel_matrix(Z, variables, alt_id_col,
                                                       'Z')

    ensure_valid_kernel_type(kernel_type, valid_kernel_list)

    # Obtain the list K with the Kernel Matrix per each alternative
    K = kernel_matrix(X,
                      alt_id_col,
                      obs_id_col,
                      variables,
                      kernel_type,
                      Z,
                      **kernel_kwargs)

    # Get the list of different alternatives in X
    alternatives = X[alt_id_col].unique()
    # Get the total number of alternatives
    n_alternatives = X[alt_id_col].nunique()

    # Create a vector which contains the column names for `K_long_format`
    columns = [obs_id_col, alt_id_col, choice_col]

    for alt in range(0, n_alternatives):
        # Get `columns` columns from X where `alt_id_col` = alt
        Obs_alt_aux = X[X[alt_id_col] == alternatives[alt]][columns]

        # Append the `K[alt]` Kernel Matrix to `K_long_format` dataframe
        if alt == 0: # First alternative
            K_long_format = K[alt]
            Obs_alt = Obs_alt_aux.copy()
        else:
            K_long_format = K_long_format.append(K[alt])
            Obs_alt = Obs_alt.append(Obs_alt_aux)

    # Insert the new columns names (those from the Kernel Matrix)
    columns = columns + list(K_long_format.columns)

    # Append obs_id_col, alt_id_col, choice_col to K
    K_long_format = np.append(Obs_alt, K_long_format, axis=1)

    # Convert K from array to dataframe
    K_long_format = pd.DataFrame(data=K_long_format,
                                 dtype=np.float64,
                                 columns=columns)

    # Convert obs_id_col, alt_id_col, choice_col columns to int
    convert_dict = {obs_id_col: int,
                    alt_id_col: int,
                    choice_col: int
                   }
    K_long_format = K_long_format.astype(convert_dict)

    # Reorder K to assure all rows for a given choice situation are contiguous
    K_long_format = K_long_format.sort_values(by=[obs_id_col, alt_id_col])

    return K_long_format


def define_kernel_specification(K_long_format,
                                alt_id_col,
                                obs_id_col,
                                specification=OrderedDict(),
                                names=OrderedDict(),
                                alpha_per_alternative=False,
                                intercept=None):
    """
    Generates the desired kernel model specification using PyLogit syntax.

    Parameters
    ----------
    K_long_format : pandas dataframe.
        A long format dataframe that contains the kernel matrix for all the
        alternatives in the dataset.
        Contains one row for each available alternative for each observation.
        Should have the specified `[obs_id_col, alt_id_col, choice_col]` column
        headings. The dtypes of all columns should be numeric.
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    obs_id_col : str.
        Denotes the column in `data` that contains the observation ID
        values for each row.
    specification : OrderedDict, optional.
        A previous specification for the model. See `specification` parameter
        on PyLogit create_choice_model function for more information. The
        recommendation is to pass an empty order dictionary by using
        OrderedDict(). Default == OrderedDict().
    names : OrderedDict, optional.
        Should have the same keys as `specification`. See `names` parameter on
        PyLogit create_choice_model function for more information. If the value
        provided in `specification` parameter is OrderedDict(), then
        `names` must be also OrderedDict(). Default == OrderedDict().
    alpha_per_alternative : True or False, optional.
        If True, then one vector of parameters to estimate is used per
        alternative.
        If False, the same vector of parameters to estimate is used for all the
        alternatives. This generates a more restricted model, but this model
        will have much less parameters to estimate.
        Default == False.
    intercept : int or None, optional.
        If None, no intercept parameter is added to the model.
        If int, an intercept parameter is added to the model. `intercept` must
        be the alternative ID of the alternative selected to be the reference
        alternative. No intercept is generated for the reference alternative
        because it is considered to be zero.
        Default == None.
    Returns
    -------
    tuple.
        Tuple composed of three element:
            - specification : OrderedDict
                Contains the specification of the kernel model.
            - names : OrderedDict
                Contains the names for the parameters to be estimated and
                defined in `specification` parameters.
            - total_vars : int
                The total number of parameters to be estimated that were created
                when the kernel model was generated.
    """
    # Check the validity of the parameters
    ensure_columns_are_in_dataframe([alt_id_col, obs_id_col],
                                    K_long_format,
                                    data_title='K_long_format')
    if not isinstance(specification, OrderedDict):
        raise ValueError("specification must be an OrderedDict.")
    if not isinstance(names, OrderedDict):
        raise ValueError("names must be an OrderedDict.")
    if not isinstance(alpha_per_alternative, bool):
        raise ValueError("alpha_per_alternative must be True or False.")
    if intercept is not None and not isinstance(intercept, int):
        raise ValueError("intercept must be an integer or None.")

    # Store the total number of variables to be estimated later
    total_vars = 0
    # List of different alternatives in K_long_format
    alternatives = K_long_format[alt_id_col].unique()

    # Determine for each alternative which observations do not contain them in
    # the choice set
    excluded_obs = {}
    for alt in alternatives:
        excluded_obs[alt] = excluded_observation_from_alternative(K_long_format,
                                                                  alt_id_col,
                                                                  obs_id_col,
                                                                  alt)

    # Add an intercept to the model specification
    if intercept is not None:
        # Check if the reference alternative is in alternatives
        if intercept not in alternatives:
            msg_1 = "intercept is not valid."
            msg_2 = "intercept should be None or one of the alternatives to " \
                     + "be use as reference."
            msg_3 = "Valid intercepts are: {}".format(alternatives)
            total_msg = "\n".join([msg_1, msg_2, msg_3])
            raise ValueError(total_msg)

        # Add intercept to specification order dict
        specification["intercept"] = list(np.delete(alternatives, \
            np.where(alternatives == intercept)))

        # Store the Alternative Specific Constant (ASC) names
        ASC_names = []
        for alt in specification["intercept"]:
            ASC_names.append('ASC {}'.format(alt))
        names["intercept"] = ASC_names
        # Increment the number of varaibles to be estimated
        total_vars = len(ASC_names)

    # Get the kernel columns in K_long_format
    K_long_format_columns_string = ''.join(K_long_format.columns)
    columns = re.findall(r"K_i_[0-9]+", K_long_format_columns_string)

    # Iterate over K_i_n, varying the observation (n)
    n = 0
    for column in columns:
        n += 1

        # Build the specification list that will be used to create the list of
        # integer or list of lists for the specification dictionary.
        specification_list = []
        names_list = []

        if alpha_per_alternative == True:
            # Generic model: there is one alpha vector per alternative
            for alt in alternatives:
                # Do not create an alpha variable for excluded observations
                # i.e. those alternatives that are not in the choice set
                if n not in excluded_obs[alt]:
                    specification_list.append(alt)
                    names_list.append('alpha_%d_%d' % (alt, n))
                    total_vars += 1
        else:
            # Restricted model: alpha vector is the same for all alternatives
            iner_specification_list = []
            for alt in alternatives:
                iner_specification_list.append(alt)
            specification_list.append(iner_specification_list)
            names_list.append('alpha_%d' % n)
            total_vars += 1

        # Add a new pair (key, value) to specification and names
        specification[column] = specification_list
        names[column] = names_list

    return([specification, names, total_vars])


def kernel_utility_function_WTP(WTP_vars,
                                WTP_point,
                                WTP_cost_column,
                                WTP_x_column,
                                alt,
                                model,
                                variables,
                                Z,
                                kernel_type,
                                scaler):
    """
    Returns the systematic utility of the selected alternative for a certain
    observation in long format using a kernel model to be used when computing
    the Willingness to Pay.

    Parameters
    ----------
    WTP_vars :
        A list with the new values for the `WTP_cost_column` and the
        `WTP_x_column` variables to be inserted on the selected alternative
        `alt` of `WTP_point`.
    WTP_point : pandas dataframe.
        A pandas dataframe in long format with only one observation (1 row per
        each alternative of that observation) for which the Willingness to Pay
        is going to be computed.
    WTP_cost_column : str
        Column name with contains the cost or price of the alternative.
    WTP_x_column : str
        Column name with contains the variable that wants to be analyzed.
    alt : int
        Indicates the alternative for which the WTP is computed.
    model : a pylogit.base_multinomial_cm_v2.MNDC_Model instance corresponding
        with the trained model.
    variables : dict.
        Python dictionary where each key represents each of the alternatives
        IDs and its associated value contains a list of characteristics from
        the long dataframe `Z` to be used on that alternative.
    Z : pandas dataframe or None, optional.
        The kernel_matrix function allows to specify a reference matrix Z_i
        to be used when computing the kernel matrix. If no Z dataframe is
        specified, then the kernel matrix is computed taking as reference
        matrix Z_i the input matrix X_i. Default == None.
    kernel_type : str, optional.
        The type of kernel function to be used. The kernel_matrix function
        relies on the scikit-learn Python package for the kernel functions.
        The value passed on this parameter specifies the scikit-learn kernel
        function which is going to be used to compute the kernel matrix.
        Default == "RBF".
    scaler : None or a scaler instance, optional.
        If none scaler was used to fit the kernel model, this parameter must be
        None. Otherwise, the an instance of the scaler function must be provided
        here. The provided scaler must contain a `transform` function which
        scale the features of the dataset according to the scaler definition.
        It is recommended to use any scaler from sklearn.preprocessing module.

    Returns
    -------
    float
        The systematic utility of the selected alternative for the observation
        provided in `ẀTP_vars` and `ẀTP_point`.
    """
    if not hasattr(scaler, 'transform'):
        raise ValueError("scaler instance must have a transform() method.")

    # Insert the new values `WTP_vars` into `WTP_point`
    WTP_point = WTP_point.copy()
    WTP_point.loc[WTP_point[model.alt_id_col] == alt,
                  [WTP_cost_column, WTP_x_column]] = WTP_vars

    # Apply scale to `WTP_point` dataframe if necessary
    if scaler is not None:
        WTP_point[[WTP_cost_column, WTP_x_column]] = scaler.transform(
            WTP_point[[WTP_cost_column, WTP_x_column]])

    # Obtain the long format kernel matrix for `WTP_point`
    K_WTP = long_format_with_kernel_matrix(WTP_point,
                                           model.alt_id_col,
                                           model.obs_id_col,
                                           model.choice_col,
                                           variables,
                                           kernel_type,
                                           Z=Z)

    # Calculate the systematic utilities for the `WTP_point`
    sys_utilities = model.get_utility(K_WTP)

    # Get all the alternatives in the model
    alts = np.unique(model.alt_IDs)

    # Return only the systematic utility of the selected alternative
    return sys_utilities[np.where(alts == alt)[0][0]]


def calculate_WTP_kernel(WTP_cost_column,
                         WTP_x_column,
                         alt,
                         model,
                         variables,
                         WTP_point,
                         Z = None,
                         kernel_type = "RBF",
                         scaler = None,
                         epsilon = None):
    """
    Calculate the willingness to pay (WTP) indicator for an observation provided
    in a long format dataframe (`WTP_point`) using a Kernel model.

    If the model contains a cost or price variable, it is possible to analyze
    the trade-off between any variable and money. It reflects the willingness
    of the decision maker to pay for a modification of another variable of the
    model.

    Parameters
    ----------
    WTP_cost_column : str
        Column name with contains the cost or price of the alternative.
    WTP_x_column : str
        Column name with contains the variable that wants to be analyzed.
    alt : int
        Indicates the alternative for which the WTP is computed.
    model : a pylogit.base_multinomial_cm_v2.MNDC_Model instance corresponding
        with the trained model.
    variables : dict.
        Python dictionary where each key represents each of the alternatives
        IDs and its associated value contains a list of characteristics from
        the long dataframe `Z` to be used on that alternative.
    WTP_point : pandas dataframe.
        A pandas dataframe in long format with only one observation (1 row per
        each alternative of that observation) for which the Willingness to Pay
        is going to be computed.
    Z : pandas dataframe or None, optional.
        The kernel_matrix function allows to specify a reference matrix Z_i
        to be used when computing the kernel matrix. If no Z dataframe is
        specified, then the kernel matrix is computed taking as reference
        matrix Z_i the input matrix X_i. Default == None.
    kernel_type : str, optional.
        The type of kernel function to be used. The kernel_matrix function
        relies on the scikit-learn Python package for the kernel functions.
        The value passed on this parameter specifies the scikit-learn kernel
        function which is going to be used to compute the kernel matrix.
        Default == "RBF".
    scaler : None or a scaler instance, optional.
        If none scaler was used to fit the kernel model, this parameter must be
        None. Otherwise, the an instance of the scaler function must be provided
        here. The provided scaler must contain a `transform` function which
        scale the features of the dataset according to the scaler definition.
        It is recommended to use any scaler from sklearn.preprocessing module.
    epsilon : array_like or None, optional.
        Increment to the objective function to use for determining the function
        gradient. If None (recommended), then the square root of the minimum
        supported float value is used.

    Returns
    -------
    float.
        The computed willingness to pay value.
    """
    # Check the validity of the parameters
    if not isinstance(model, base_multinomial_cm_v2.MNDC_Model):
        raise ValueError("model must be an instance of MNDC_Model class.")

    ensure_columns_are_in_dataframe([model.obs_id_col,
                                     model.alt_id_col,
                                     model.choice_col,
                                     WTP_cost_column,
                                     WTP_x_column],
                                    WTP_point,
                                    data_title='WTP_point')

    if epsilon is not None and not isinstance(epsilon, float):
        raise ValueError("epsilon must be None or float.")

    if not hasattr(scaler, 'transform'):
        raise ValueError("scaler instance must have a transform() method.")

    # Set `WTP_vars`, which is a list with the values of the `WTP_cost_column`
    # and the `WTP_x_column` variables of the alternative `alt` from the
    # selected `WTP_point`.
    # `WTP_vars` is going to be used to compute the numerical derivatives.
    WTP_vars = WTP_point.loc[WTP_point[model.alt_id_col] == alt,
                             [WTP_cost_column, WTP_x_column]].values[0]

    # Set the `WTP_vars` increment use for approximating the function gradient
    if epsilon is None:
        epsilon = np.sqrt(np.finfo(float).eps)

    # Get the gradient vector of the utility function for the variables provided
    # in `WTP_cost_column` and `WTP_x_column`
    grad = optimize.approx_fprime(WTP_vars,
                                  kernel_utility_function_WTP,
                                  epsilon,
                                  WTP_point,
                                  WTP_cost_column,
                                  WTP_x_column,
                                  alt,
                                  model,
                                  variables,
                                  Z,
                                  kernel_type,
                                  scaler)

    try:
        # Calculate the WTP value: WTP = - gradient(x)/gradient(cost)
        WTP = -grad[1]/grad[0]
    except ZeroDivisionError:
        # Check if a division by zero was attempted
        msg_1 = "Error! A division by zero occurred when computing WTP value."+\
                " This means that the gradient of the cost variable was 0 or "+\
                "maybe close to zero."
        msg_2 = "Please, check the WTP_point provided or try a different one. "
        msg_3 = "You can also try a different value for epsilon variable."
        total_msg = "\n".join([msg_1, msg_2, msg_3])
        raise ValueError(total_msg)

    return WTP
