"""
Helpers for numerical derivatives, integration and root finding.
"""

# Standard imports
import logging
import typing

# Third party imports
import numpy as np

# Local imports
from crayon.shared.types import Array, BooleanArray, FloatArray, NumericType

logger = logging.getLogger(__name__)

STENCIL_FIRST_DERIVATIVE = {
    2: ((-1, -1 / 2), (1, 1 / 2)),
    4: ((-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12)),
}

STENCIL_SECOND_DERIVATIVE = {
    2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
    4: ((-2, -1 / 12), (-1, 4 / 3), (0, -5 / 2), (1, 4 / 3), (2, -1 / 12)),
}

STENCIL_MIXED_SECOND_DERIVATIVE = {
    2: ((1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25)),
    4: (
        (1, -2, -63 / 600),
        (2, -1, -63 / 600),
        (-2, 1, -63 / 600),
        (-1, 2, -63 / 600),
        (-1, -2, 63 / 600),
        (-2, -1, 63 / 600),
        (1, 2, 63 / 600),
        (2, 1, 63 / 600),
        (2, -2, 44 / 600),
        (-2, 2, 44 / 600),
        (-2, -2, -44 / 600),
        (2, 2, -44 / 600),
        (-1, -1, 74 / 600),
        (1, 1, 74 / 600),
        (1, -1, -74 / 600),
        (-1, 1, -74 / 600),
    ),
}

STENCIL_THIRD_DERIVATIVE = {
    2: ((-2, -1 / 2), (-1, 1.0), (1, -1.0), (2, 1 / 2)),
    4: (
        (-3, 1 / 8),
        (-2, -1),
        (-1, 13 / 8),
        (1, -13 / 8),
        (2, 1),
        (3, -1 / 8),
    ),
}

STENCIL_MIXED_THIRD_DERIVATIVE = {}


def first_derivative_finite_difference(
    argument: FloatArray,
    value_function: typing.Callable[[FloatArray], Array],
    value_shape: tuple[int],
    /,
    *,
    h: float = 1e-6,
    order: int = 2,
    is_complex: bool = False,
) -> Array:
    """
    Calculate first derivative of a function using finite differences.

    Parameters
    ----------
    argument : np.array[float]
        Argument of value_function to evaluate derivative at.
    value_function : callable[[np.array[float]], np.array[float]]
        Function which takes array like argument and returns value with
        shape value_shape.
    value_shape : tuple[int]
        Shape of value_function output.
    h : float, optional
        Finite difference step size.
    order : int, optional
        Order of finite difference.
    is_complex : bool, optional
        If True, value_function is complex.

    Returns
    -------
    derivative : np.array
        First derivative of value_function at argument.
    """
    argument = np.asarray(argument)
    n = argument.size
    scalar = n == 1

    if scalar:
        argument = argument.reshape(1)

    if order in STENCIL_FIRST_DERIVATIVE:
        stencil = STENCIL_FIRST_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"First derivative order = {order}")

    if is_complex:
        derivative = np.zeros((*value_shape, n), dtype=complex)
    else:
        derivative = np.zeros((*value_shape, n))

    dummy_arg = np.zeros_like(argument)
    for i in range(n):
        dummy_arg[...] = argument
        for step, weight in stencil:
            dummy_arg[i] = argument[i] + step * h
            derivative[..., i] += weight * value_function(dummy_arg)

    return derivative / h


def second_derivative_finite_difference(
    argument: FloatArray,
    value_function: typing.Callable[[FloatArray], Array],
    value_shape: tuple[int],
    /,
    *,
    h: float = 1e-4,
    order: int = 2,
    is_complex: bool = False,
):
    """
    Calculate second derivative of a function using finite differences.

    Parameters
    ----------
    argument : np.array[float]
        Argument of value_function to evaluate derivative at.
    value_function : callable[[np.array[float]], np.array[float]]
        Function which takes array like argument and returns value with
        shape value_shape.
    value_shape : tuple[int]
        Shape of value_function output.
    h : float, optional
        Finite difference step size.
    order : int, optional
        Order of finite difference.
    is_complex : bool, optional
        If True, value_function is complex.

    Returns
    -------
    derivative : np.array
        Second derivative of value_function at argument.
    """
    argument = np.asarray(argument)
    n = argument.size
    scalar = n == 1

    if scalar:
        argument = argument.reshape(1)

    if order in STENCIL_SECOND_DERIVATIVE:
        stencil = STENCIL_SECOND_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"Second derivative order = {order}")

    if order in STENCIL_MIXED_SECOND_DERIVATIVE:
        stencil_mixed = STENCIL_MIXED_SECOND_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"Second mixed derivative order = {order}")

    if is_complex:
        derivative = np.zeros((*value_shape, n, n), dtype=complex)
    else:
        derivative = np.zeros((*value_shape, n, n))

    dummy_arg = np.zeros_like(argument)

    # Non-mixed derivatives.
    for i in range(n):
        dummy_arg[...] = argument
        for step, weight in stencil:
            dummy_arg[i] = argument[i] + step * h
            derivative[..., i, i] += weight * value_function(dummy_arg)

    # Mixed derivatives.
    for i in range(n):
        for j in range(i + 1, n):
            dummy_arg[...] = argument
            for step1, step2, weight in stencil_mixed:
                dummy_arg[i] = argument[i] + step1 * h
                dummy_arg[j] = argument[j] + step2 * h
                derivative[..., i, j] += weight * value_function(dummy_arg)

            # Second derivatives commute.
            derivative[..., j, i] = derivative[..., i, j]

    return derivative / h**2


def second_mixed_derivative_finite_difference(
    argument_1: FloatArray,
    argument_2: FloatArray,
    value_function: typing.Callable[[FloatArray, FloatArray], Array],
    value_shape: tuple[int],
    /,
    *,
    h: float = 1e-4,
    order: int = 2,
    is_complex: bool = False,
):
    """
    Calculate mixed second derivative of a function using finite differences.

    Parameters
    ----------
    argument_1 : np.array[float]
        First argument of value_function to evaluate derivative at.
    value_function : callable[
        [np.array[float], np.array[float]], np.array[float]
    ]
        Function which takes 2 array like arguments and returns value with
        shape value_shape.
    value_shape : tuple[int]
        Shape of value_function output.
    h : float, optional
        Finite difference step size.
    order : int, optional
        Order of finite difference.
    is_complex : bool, optional
        If True, value_function is complex.

    Returns
    -------
    derivative : np.array
        Second mixed derivative of value_function at arguments.
    """
    argument_1, argument_2 = np.asarray(argument_1), np.asarray(argument_2)
    n1, n2 = argument_1.size, argument_2.size
    scalar_1, scalar_2 = n1 == 1, n2 == 1

    if scalar_1:
        argument_1 = argument_1.reshape(1)
    if scalar_2:
        argument_2 = argument_2.reshape(1)

    if order in STENCIL_MIXED_SECOND_DERIVATIVE:
        stencil_mixed = STENCIL_MIXED_SECOND_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"Second mixed derivative order = {order}")

    if is_complex:
        derivative = np.zeros((*value_shape, n1, n2), dtype=complex)
    else:
        derivative = np.zeros((*value_shape, n1, n2))

    dummy_arg_1 = np.zeros_like(argument_1)
    dummy_arg_2 = np.zeros_like(argument_2)
    for i in range(n1):
        for j in range(n2):
            dummy_arg_1[...] = argument_1
            dummy_arg_2[...] = argument_2

            for step1, step2, weight in stencil_mixed:
                dummy_arg_1[i] = argument_1[i] + step1 * h
                dummy_arg_2[j] = argument_2[j] + step2 * h
                derivative[..., i, j] += weight * value_function(
                    dummy_arg_1, dummy_arg_2
                )

    return derivative / h**2


def third_derivative_finite_difference(
    argument: FloatArray,
    value_function: typing.Callable,
    value_shape: tuple[int],
    /,
    *,
    h: float = 1e-3,
    order: int = 2,
    is_complex: bool = False,
):
    """
    Calculate third derivative of a function using finite differences.

    Parameters
    ----------
    argument : np.array[float]
        Argument of value_function to evaluate derivative at.
    value_function : callable[[np.array[float]], np.array[float]]
        Function which takes array like argument and returns value with
        shape value_shape.
    value_shape : tuple[int]
        Shape of value_function output.
    h : float, optional
        Finite difference step size.
    order : int, optional
        Order of finite difference.
    is_complex : bool, optional
        If True, value_function is complex.

    Returns
    -------
    derivative : np.array
        Third derivative of value_function at argument.
    """
    argument = np.asarray(argument)
    n = argument.size
    scalar = n == 1

    if scalar:
        argument = argument.reshape(1)

    if order in STENCIL_THIRD_DERIVATIVE:
        stencil = STENCIL_THIRD_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"Third derivative order = {order}")

    if order in STENCIL_MIXED_THIRD_DERIVATIVE:
        stencil_mixed = STENCIL_MIXED_THIRD_DERIVATIVE[order]
    else:
        raise NotImplementedError(f"Third mixed derivative order = {order}")

    if is_complex:
        derivative = np.zeros((*value_shape, n, n), dtype=complex)
    else:
        derivative = np.zeros((*value_shape, n, n))

    dummy_arg = np.zeros_like(argument)

    # Non-mixed derivatives.
    for i in range(n):
        dummy_arg[...] = argument
        for step, weight in stencil:
            dummy_arg[i] = argument[i] + step * h
            derivative[..., i, i, i] += weight * value_function(dummy_arg)

    # Mixed derivatives.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dummy_arg[...] = argument
                for step1, step2, step3, weight in stencil_mixed:
                    dummy_arg[i] = argument[i] + step1 * h
                    dummy_arg[j] = argument[j] + step2 * h
                    dummy_arg[k] = argument[k] + step3 * h

                    derivative[..., i, j, k] += weight * value_function(
                        dummy_arg
                    )

                # Third derivatives commute.
                derivative[..., j, i, k] = derivative[..., i, j, k]
                derivative[..., k, j, i] = derivative[..., i, j, k]
                derivative[..., i, k, j] = derivative[..., i, j, k]

    return derivative / h**3


# Sample points and weights for 24 node Gaussian quadrature.
_sample_points_leg24, _weights_leg24 = np.polynomial.legendre.leggauss(24)


def get_leggauss_samples_weights(x0: float, x1: float, /, *, n: int = 24):
    r"""
    Return sample values and weights for n point Gauss-Legendre quadrature for
    an interval [x0, x1].

    Parameters
    ----------
    x0, x1 : float
        Start and end of integration interval
    n : int
        Number of nodes. Default is 24.

    Returns
    -------
    nodes : np.array[float]
        Integration nodes.
    weights : np.array[float]
        Integration sample weights.

    Notes
    -----
    Standard gaussian quadrature takes \int f(x) dx = \sum_i [w_i f(x_i)]
    where x in [-1, 1] and x_i, w_i are the sample points and weights.

    We can map for x in [x0, x1] using the mapping

    x_i -> x0 + 0.5 * (x1 - x0) * (1 + x_i)
    w_i -> 0.5 * (x1 - x0) * w_i
    """
    # Get sample points and weights on interval [-1, 1].
    if n == 24:  # noqa: PLR2004
        samples, weights = _sample_points_leg24, _weights_leg24
    else:
        samples, weights = np.polynomial.legendre.leggauss(n)

    # Map sample points and weights to interval [x0, x1].
    samples = x0 + 0.5 * (x1 - x0) * (samples + 1)
    weights = 0.5 * (x1 - x0) * weights

    return samples, weights


# Taken from https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
_gauss_kronrod_7_15 = (
    # Nodes.
    np.array([
        -0.9914553711208120,
        -0.9491079123427580,
        -0.8648644233597690,
        -0.7415311855993940,
        -0.5860872354676910,
        -0.4058451513773970,
        -0.2077849550078980,
        0.0000000000000000,
        0.2077849550078980,
        0.4058451513773970,
        0.5860872354676910,
        0.7415311855993940,
        0.8648644233597690,
        0.9491079123427580,
        0.9914553711208120,
    ]),
    # Kronrod weights.
    np.array([
        0.0229353220105292,
        0.0630920926299785,
        0.1047900103222500,
        0.1406532597155250,
        0.1690047266392670,
        0.1903505780647850,
        0.2044329400752980,
        0.2094821410847270,
        0.2044329400752980,
        0.1903505780647850,
        0.1690047266392670,
        0.1406532597155250,
        0.1047900103222500,
        0.0630920926299785,
        0.0229353220105292,
    ]),
    # Gauss mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.1294849661688690,
        0.2797053914892760,
        0.3818300505051180,
        0.4179591836734690,
        0.3818300505051180,
        0.2797053914892760,
        0.1294849661688690,
    ]),
)

_gauss_kronrod_10_21 = (
    # Nodes.
    np.array([
        -0.9956571630258080,
        -0.9739065285171710,
        -0.9301574913557080,
        -0.8650633666889840,
        -0.7808177265864160,
        -0.6794095682990240,
        -0.5627571346686040,
        -0.4333953941292470,
        -0.2943928627014600,
        -0.1488743389816310,
        0.0000000000000000,
        0.1488743389816310,
        0.2943928627014600,
        0.4333953941292470,
        0.5627571346686040,
        0.6794095682990240,
        0.7808177265864160,
        0.8650633666889840,
        0.9301574913557080,
        0.9739065285171710,
        0.9956571630258080,
    ]),
    # Weights.
    np.array([
        0.0116946388673718,
        0.0325581623079647,
        0.0547558965743519,
        0.0750396748109199,
        0.0931254545836976,
        0.1093871588022970,
        0.1234919762620650,
        0.1347092173114730,
        0.1427759385770600,
        0.1477391049013380,
        0.1494455540029160,
        0.1477391049013380,
        0.1427759385770600,
        0.1347092173114730,
        0.1234919762620650,
        0.1093871588022970,
        0.0931254545836976,
        0.0750396748109199,
        0.0547558965743519,
        0.0325581623079647,
        0.0116946388673718,
    ]),
    # Mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.0666713443086881,
        0.1494513491505800,
        0.2190863625159820,
        0.2692667193099960,
        0.2955242247147520,
        0.2955242247147520,
        0.2692667193099960,
        0.2190863625159820,
        0.1494513491505800,
        0.0666713443086881,
    ]),
)

_gauss_kronrod_15_31 = (
    # Nodes.
    np.array([
        -0.9980022986933970,
        -0.9879925180204850,
        -0.9677390756791390,
        -0.9372733924007050,
        -0.8972645323440810,
        -0.8482065834104270,
        -0.7904185014424650,
        -0.7244177313601700,
        -0.6509967412974160,
        -0.5709721726085380,
        -0.4850818636402390,
        -0.3941513470775630,
        -0.2991800071531680,
        -0.2011940939974340,
        -0.1011420669187170,
        0.0000000000000000,
        0.1011420669187170,
        0.2011940939974340,
        0.2991800071531680,
        0.3941513470775630,
        0.4850818636402390,
        0.5709721726085380,
        0.6509967412974160,
        0.7244177313601700,
        0.7904185014424650,
        0.8482065834104270,
        0.8972645323440810,
        0.9372733924007050,
        0.9677390756791390,
        0.9879925180204850,
        0.9980022986933970,
    ]),
    # Weights.
    np.array([
        0.0053774798729233,
        0.0150079473293161,
        0.0254608473267153,
        0.0353463607913758,
        0.0445897513247648,
        0.0534815246909280,
        0.0620095678006706,
        0.0698541213187282,
        0.0768496807577203,
        0.0830805028231330,
        0.0885644430562117,
        0.0931265981708253,
        0.0966427269836236,
        0.0991735987217919,
        0.1007698455238750,
        0.1013300070147910,
        0.1007698455238750,
        0.0991735987217919,
        0.0966427269836236,
        0.0931265981708253,
        0.0885644430562117,
        0.0830805028231330,
        0.0768496807577203,
        0.0698541213187282,
        0.0620095678006706,
        0.0534815246909280,
        0.0445897513247648,
        0.0353463607913758,
        0.0254608473267153,
        0.0150079473293161,
        0.0053774798729233,
    ]),
    # Mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.0307532419961172,
        0.0703660474881081,
        0.1071592204671710,
        0.1395706779261540,
        0.1662692058169930,
        0.1861610000155620,
        0.1984314853271110,
        0.2025782419255610,
        0.1984314853271110,
        0.1861610000155620,
        0.1662692058169930,
        0.1395706779261540,
        0.1071592204671710,
        0.0703660474881081,
        0.0307532419961172,
    ]),
)

_gauss_kronrod_20_41 = (
    # Nodes.
    np.array([
        -0.9988590315882770,
        -0.9931285991850940,
        -0.9815078774502500,
        -0.9639719272779130,
        -0.9408226338317540,
        -0.9122344282513250,
        -0.8782768112522810,
        -0.8391169718222180,
        -0.7950414288375510,
        -0.7463319064601500,
        -0.6932376563347510,
        -0.6360536807265150,
        -0.5751404468197100,
        -0.5108670019508270,
        -0.4435931752387250,
        -0.3737060887154190,
        -0.3016278681149130,
        -0.2277858511416450,
        -0.1526054652409220,
        -0.0765265211334973,
        0.0000000000000000,
        0.0765265211334973,
        0.1526054652409220,
        0.2277858511416450,
        0.3016278681149130,
        0.3737060887154190,
        0.4435931752387250,
        0.5108670019508270,
        0.5751404468197100,
        0.6360536807265150,
        0.6932376563347510,
        0.7463319064601500,
        0.7950414288375510,
        0.8391169718222180,
        0.8782768112522810,
        0.9122344282513250,
        0.9408226338317540,
        0.9639719272779130,
        0.9815078774502500,
        0.9931285991850940,
        0.9988590315882770,
    ]),
    # Weights.
    np.array([
        0.0030735837185205,
        0.0086002698556429,
        0.0146261692569712,
        0.0203883734612665,
        0.0258821336049511,
        0.0312873067770327,
        0.0366001697582007,
        0.0416688733279736,
        0.0464348218674976,
        0.0509445739237286,
        0.0551951053482859,
        0.0591114008806395,
        0.0626532375547811,
        0.0658345971336184,
        0.0686486729285216,
        0.0710544235534440,
        0.0730306903327866,
        0.0745828754004991,
        0.0757044976845566,
        0.0763778676720807,
        0.0766007119179996,
        0.0763778676720807,
        0.0757044976845566,
        0.0745828754004991,
        0.0730306903327866,
        0.0710544235534440,
        0.0686486729285216,
        0.0658345971336184,
        0.0626532375547811,
        0.0591114008806395,
        0.0551951053482859,
        0.0509445739237286,
        0.0464348218674976,
        0.0416688733279736,
        0.0366001697582007,
        0.0312873067770327,
        0.0258821336049511,
        0.0203883734612665,
        0.0146261692569712,
        0.0086002698556429,
        0.0030735837185205,
    ]),
    # Mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.0176140071391521,
        0.0406014298003869,
        0.0626720483341090,
        0.0832767415767047,
        0.1019301198172400,
        0.1181945319615180,
        0.1316886384491760,
        0.1420961093183820,
        0.1491729864726030,
        0.1527533871307250,
        0.1527533871307250,
        0.1491729864726030,
        0.1420961093183820,
        0.1316886384491760,
        0.1181945319615180,
        0.1019301198172400,
        0.0832767415767047,
        0.0626720483341090,
        0.0406014298003869,
        0.0176140071391521,
    ]),
)

_gauss_kronrod_25_51 = (
    # Nodes.
    np.array([
        -0.9992621049926090,
        -0.9955569697904980,
        -0.9880357945340770,
        -0.9766639214595170,
        -0.9616149864258420,
        -0.9429745712289740,
        -0.9207471152817010,
        -0.8949919978782750,
        -0.8658470652932750,
        -0.8334426287608340,
        -0.7978737979985000,
        -0.7592592630373570,
        -0.7177664068130840,
        -0.6735663684734680,
        -0.6268100990103170,
        -0.5776629302412220,
        -0.5263252843347190,
        -0.4730027314457140,
        -0.4178853821930370,
        -0.3611723058093870,
        -0.3030895389311070,
        -0.2438668837209880,
        -0.1837189394210480,
        -0.1228646926107100,
        -0.0615444830056850,
        0.0000000000000000,
        0.0615444830056850,
        0.1228646926107100,
        0.1837189394210480,
        0.2438668837209880,
        0.3030895389311070,
        0.3611723058093870,
        0.4178853821930370,
        0.4730027314457140,
        0.5263252843347190,
        0.5776629302412220,
        0.6268100990103170,
        0.6735663684734680,
        0.7177664068130840,
        0.7592592630373570,
        0.7978737979985000,
        0.8334426287608340,
        0.8658470652932750,
        0.8949919978782750,
        0.9207471152817010,
        0.9429745712289740,
        0.9616149864258420,
        0.9766639214595170,
        0.9880357945340770,
        0.9955569697904980,
        0.9992621049926090,
    ]),
    # Weights.
    np.array([
        0.0019873838923303,
        0.0055619321353567,
        0.0094739733861742,
        0.0132362291955716,
        0.0168478177091282,
        0.0204353711458828,
        0.0240099456069532,
        0.0274753175878517,
        0.0307923001673874,
        0.0340021302743293,
        0.0371162714834155,
        0.0400838255040323,
        0.0428728450201700,
        0.0455029130499217,
        0.0479825371388367,
        0.0502776790807156,
        0.0523628858064074,
        0.0542511298885454,
        0.0559508112204123,
        0.0574371163615678,
        0.0586896800223942,
        0.0597203403241740,
        0.0605394553760458,
        0.0611285097170530,
        0.0614711898714253,
        0.0615808180678329,
        0.0614711898714253,
        0.0611285097170530,
        0.0605394553760458,
        0.0597203403241740,
        0.0586896800223942,
        0.0574371163615678,
        0.0559508112204123,
        0.0542511298885454,
        0.0523628858064074,
        0.0502776790807156,
        0.0479825371388367,
        0.0455029130499217,
        0.0428728450201700,
        0.0400838255040323,
        0.0371162714834155,
        0.0340021302743293,
        0.0307923001673874,
        0.0274753175878517,
        0.0240099456069532,
        0.0204353711458828,
        0.0168478177091282,
        0.0132362291955716,
        0.0094739733861742,
        0.0055619321353567,
        0.0019873838923303,
    ]),
    # Mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.0113937985010262,
        0.0263549866150321,
        0.0409391567013063,
        0.0549046959758351,
        0.0680383338123569,
        0.0801407003350010,
        0.0910282619829636,
        0.1005359490670500,
        0.1085196244742630,
        0.1148582591457110,
        0.1194557635357840,
        0.1222424429903100,
        0.1231760537267150,
        0.1222424429903100,
        0.1194557635357840,
        0.1148582591457110,
        0.1085196244742630,
        0.1005359490670500,
        0.0910282619829636,
        0.0801407003350010,
        0.0680383338123569,
        0.0549046959758351,
        0.0409391567013063,
        0.0263549866150321,
        0.0113937985010262,
    ]),
)

_gauss_kronrod_30_61 = (
    # Nodes.
    np.array([
        -0.9994844100504900,
        -0.9968934840746490,
        -0.9916309968704040,
        -0.9836681232797470,
        -0.9731163225011260,
        -0.9600218649683070,
        -0.9443744447485590,
        -0.9262000474292740,
        -0.9055733076999070,
        -0.8825605357920520,
        -0.8572052335460610,
        -0.8295657623827680,
        -0.7997278358218390,
        -0.7677774321048260,
        -0.7337900624532260,
        -0.6978504947933150,
        -0.6600610641266260,
        -0.6205261829892420,
        -0.5793452358263610,
        -0.5366241481420190,
        -0.4924804678617780,
        -0.4470337695380890,
        -0.4004012548303940,
        -0.3527047255308780,
        -0.3040732022736250,
        -0.2546369261678890,
        -0.2045251166823090,
        -0.1538699136085830,
        -0.1028069379667370,
        -0.0514718425553176,
        0.0000000000000000,
        0.0514718425553176,
        0.1028069379667370,
        0.1538699136085830,
        0.2045251166823090,
        0.2546369261678890,
        0.3040732022736250,
        0.3527047255308780,
        0.4004012548303940,
        0.4470337695380890,
        0.4924804678617780,
        0.5366241481420190,
        0.5793452358263610,
        0.6205261829892420,
        0.6600610641266260,
        0.6978504947933150,
        0.7337900624532260,
        0.7677774321048260,
        0.7997278358218390,
        0.8295657623827680,
        0.8572052335460610,
        0.8825605357920520,
        0.9055733076999070,
        0.9262000474292740,
        0.9443744447485590,
        0.9600218649683070,
        0.9731163225011260,
        0.9836681232797470,
        0.9916309968704040,
        0.9968934840746490,
        0.9994844100504900,
    ]),
    # Weights.
    np.array([
        0.0013890136986770,
        0.0038904611270999,
        0.0066307039159313,
        0.0092732796595178,
        0.0118230152534963,
        0.0143697295070458,
        0.0169208891890532,
        0.0194141411939423,
        0.0218280358216091,
        0.0241911620780806,
        0.0265099548823331,
        0.0287540487650412,
        0.0309072575623877,
        0.0329814470574837,
        0.0349793380280600,
        0.0368823646518212,
        0.0386789456247275,
        0.0403745389515359,
        0.0419698102151642,
        0.0434525397013560,
        0.0448148001331626,
        0.0460592382710069,
        0.0471855465692991,
        0.0481858617570871,
        0.0490554345550297,
        0.0497956834270742,
        0.0504059214027823,
        0.0508817958987496,
        0.0512215478492587,
        0.0514261285374590,
        0.0514947294294515,
        0.0514261285374590,
        0.0512215478492587,
        0.0508817958987496,
        0.0504059214027823,
        0.0497956834270742,
        0.0490554345550297,
        0.0481858617570871,
        0.0471855465692991,
        0.0460592382710069,
        0.0448148001331626,
        0.0434525397013560,
        0.0419698102151642,
        0.0403745389515359,
        0.0386789456247275,
        0.0368823646518212,
        0.0349793380280600,
        0.0329814470574837,
        0.0309072575623877,
        0.0287540487650412,
        0.0265099548823331,
        0.0241911620780806,
        0.0218280358216091,
        0.0194141411939423,
        0.0169208891890532,
        0.0143697295070458,
        0.0118230152534963,
        0.0092732796595178,
        0.0066307039159313,
        0.0038904611270999,
        0.0013890136986770,
    ]),
    # Mask.
    np.array([
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]),
    # Gauss weights.
    np.array([
        0.0079681924961666,
        0.0184664683110909,
        0.0287847078833233,
        0.0387991925696270,
        0.0484026728305940,
        0.0574931562176190,
        0.0659742298821804,
        0.0737559747377052,
        0.0807558952294202,
        0.0868997872010829,
        0.0921225222377861,
        0.0963687371746442,
        0.0995934205867952,
        0.1017623897484050,
        0.1028526528935580,
        0.1028526528935580,
        0.1017623897484050,
        0.0995934205867952,
        0.0963687371746442,
        0.0921225222377861,
        0.0868997872010829,
        0.0807558952294202,
        0.0737559747377052,
        0.0659742298821804,
        0.0574931562176190,
        0.0484026728305940,
        0.0387991925696270,
        0.0287847078833233,
        0.0184664683110909,
        0.0079681924961666,
    ]),
)

_gauss_kronrod = {
    7: _gauss_kronrod_7_15,
    10: _gauss_kronrod_10_21,
    15: _gauss_kronrod_15_31,
    20: _gauss_kronrod_20_41,
    25: _gauss_kronrod_25_51,
    30: _gauss_kronrod_30_61,
}


def gauss_kronrod_nodes_weights(
    x0: float, x1: float, n: int
) -> tuple[FloatArray, FloatArray, BooleanArray, FloatArray]:
    """
    Get Gauss-Kronrod nodes and weights for interval [x0, x1].

    Parameters
    ----------
    x0, x1 : float
        Start and end value of integration.
    n : int
        Number of samples. Must be 7, 10, 15, 20, 25 or 30.

    Returns
    -------
    nodes : np.array[float]
        Samples.
    weights : np.array[float]
        Weights.
    mask_gauss : np.array[bool]
        Mask for nodes used for Gaussian quadrature.
    weights_gauss : np.array[float]
        Weights for Gaussian quadrature.

    Raises
    ------
    ValueError
        n not in set.
    """
    if n in _gauss_kronrod:
        nodes, weights, mask_gauss, weights_gauss = _gauss_kronrod[n]
    else:
        raise ValueError(f"Invalid n: {n}")

    dx = x1 - x0

    return (
        0.5 * (x0 + x1 + dx * nodes),
        0.5 * dx * weights,
        mask_gauss,
        0.5 * dx * weights_gauss,
    )


def solve_quadratic(
    a: NumericType, b: NumericType, c: NumericType
) -> tuple[NumericType, NumericType]:
    """
    Return both roots of a quadratic equation ax^2 + bx + c = 0.

    Parameters
    ----------
    a, b, c : float
        Quadratic coefficients.

    Returns
    -------
    y_plus, y_minus : float
        Both solutions.

    Notes
    -----
    The first root is the '+' solution in the classical quadratic formula
    while the second root is the '-' solution.
    """
    abs_a, abs_b, abs_c = abs(a), abs(b), abs(c)

    if abs_a == 0:
        if abs_b == 0 and abs_c == 0:
            # All coefficients are zero.
            logger.debug("Equation is zero.")
            return 0, 0
        logger.debug("Linear equation.")
        # Linear equation with only 1 root.
        x_plus = -c / b
        return x_plus, x_plus

    # If the coefficients are extremely large b^2 - 4ac may lose precision.
    coeff_min, coeff_max = min(abs_a, abs_b, abs_c), max(abs_a, abs_b, abs_c)
    if 1e8 * coeff_min < coeff_max:
        # The coefficients are very far apart in magnitude so attempting
        # to normalise them will likely cause loss of accuracy.
        logger.debug("Cannot normalise coefficients.")
    else:
        # Normalise coefficients to bring them to order unity.
        logger.debug("Normalise coefficients.")
        a, b, c = a / coeff_max, b / coeff_max, c / coeff_max

    # If b^2 >> 4ac, plus root is not well calculated due to round off error.
    descriminant_sqrt = np.sqrt(b * b - 4 * a * c)

    if descriminant_sqrt == 0:
        # If descriminant is zero there is a shared root.
        x_plus = -b / (2 * a)
        x_minus = x_plus
    elif 1e8 * abs(-b + descriminant_sqrt) < 1:
        # Bad round off error expected from subtracting two ~= values.
        # Instead add them and use alternative formula.
        logger.debug("Descriminant imbalance.")
        x_plus = 2 * c / (-b - descriminant_sqrt)
        x_minus = (-b - descriminant_sqrt) / (2 * a)
    else:
        x_plus = (-b + descriminant_sqrt) / (2 * a)
        x_minus = 2 * c / (-b + descriminant_sqrt)

    return x_plus, x_minus
