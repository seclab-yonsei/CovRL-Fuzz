from enum import Enum


class ErrorType(Enum):
    SYNTAX_ERROR = "syntaxError"
    REFERENCE_ERROR = "referenceError"
    TYPE_ERROR = "typeError"
    RANGE_ERROR = "rangeError"
    URI_ERROR = "uriError"
    INTERNAL_ERROR = "internalError"


V8_ERROR = {
    "SyntaxError:": ErrorType.SYNTAX_ERROR.value,
    "ReferenceError:": ErrorType.REFERENCE_ERROR.value,
    "TypeError:": ErrorType.TYPE_ERROR.value,
    "RangeError:": ErrorType.RANGE_ERROR.value,
    "URIError:": ErrorType.URI_ERROR.value,
    "Error loading file": ErrorType.INTERNAL_ERROR.value,
    "Error executing file": ErrorType.INTERNAL_ERROR.value,
}

JSC_ERROR = {
    "Exception: SyntaxError:": ErrorType.SYNTAX_ERROR.value,
    "Exception: ReferenceError:": ErrorType.REFERENCE_ERROR.value,
    "Exception: TypeError:": ErrorType.TYPE_ERROR.value,
    "Exception: RangeError:": ErrorType.RANGE_ERROR.value,
    "Exception: URIError:": ErrorType.URI_ERROR.value,
    "Could not open file:": ErrorType.INTERNAL_ERROR.value,
}

CHAKRA_ERROR = {
    "SyntaxError:": ErrorType.SYNTAX_ERROR.value,
    "ReferenceError:": ErrorType.REFERENCE_ERROR.value,
    "TypeError:": ErrorType.TYPE_ERROR.value,
    "RangeError:": ErrorType.RANGE_ERROR.value,
    "URIError:": ErrorType.URI_ERROR.value,
    "Error in opening file:": ErrorType.INTERNAL_ERROR.value,
}

JERRY_ERROR = {
    "SyntaxError": ErrorType.SYNTAX_ERROR.value,
    "ReferenceError": ErrorType.REFERENCE_ERROR.value,
    "TypeError": ErrorType.TYPE_ERROR.value,
    "RangeError": ErrorType.RANGE_ERROR.value,
    "URIError": ErrorType.URI_ERROR.value,
    "Unhandled exception": ErrorType.INTERNAL_ERROR.value,
}


ERROR_MAPPINGS = {
    "v8": V8_ERROR,
    "jsc": JSC_ERROR,
    "chakra": CHAKRA_ERROR,
    "jerry": JERRY_ERROR,
}


def map_target_error(target_arg="v8"):
    return ERROR_MAPPINGS.get(target_arg.lower())
