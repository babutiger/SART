import sympy as sp
import re

def combine_and_simplify(expression):
    """Simplify an expression string and infer the sign of each symbolic variable."""
    variable_pattern = r'x\d+_\d+(?:_[a-zA-Z_]+)?'
    variables = set(re.findall(variable_pattern, expression))

    if not variables:
        simplified_expr = sp.simplify(expression)
        return str(simplified_expr), {}

    sympy_vars = sp.symbols(' '.join(variables))

    if isinstance(sympy_vars, sp.Symbol):
        var_dict = {str(sympy_vars): sympy_vars}
    else:
        var_dict = {str(var): var for var in sympy_vars}

    sympy_expr = sp.sympify(expression, locals=var_dict)
    simplified_expr = sp.simplify(sympy_expr)

    if isinstance(sympy_vars, sp.Symbol):
        coefficients = {str(sympy_vars): simplified_expr.coeff(sympy_vars)}
    else:
        coefficients = {str(var): simplified_expr.coeff(var) for var in sympy_vars}

    signs = {var: '+' if coefficients[var] > 0 else '-' for var in coefficients}
    return str(simplified_expr), signs


if __name__== '__main__':
    expression = '1*(1*(0.5*(1*x0_0+1*x0_1+0)+1.0)+1*(0.5*(1*x0_0+(-1)*x0_1+0)+1.0)+0)+1*(1*(0.5*(1*x0_0+1*x0_1+0)+1.0)+(-1)*(0)+0)+1'
    combined_expression, variable_signs = combine_and_simplify(expression)
    print("Combined Expression:", combined_expression)
    print("Variable Signs:", variable_signs)
