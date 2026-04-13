import re

from sart.layerabs.support.lstr2str import combine_and_simplify
from sart.layerabs.support.exnum import extract_numbers
from sart.layerabs.support.isconstant import is_constant



def replace_sub(expression, variable, replacement):
    variable_pattern = rf'\b{re.escape(variable)}\b'
    replaced_expression = re.sub(variable_pattern, replacement, expression)
    return replaced_expression



if __name__== '__main__':
    t_deeppoly_low = "- 0.0109756*x1_7 - 0.016610429422520177*x1_70 - 0.0007966489329466094*x1_72 - 0.127605*x1_79 "
    key = 'x1_70'
    replacement = 'hhahaha'

    replaced_expression = replace_sub(t_deeppoly_low, key, replacement)
    print(replaced_expression)
