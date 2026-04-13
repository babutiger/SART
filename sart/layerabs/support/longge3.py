def add(left, right):
    if right[0] == '-':
        return_str = left + right
        del(left)
        del (right)
        return return_str

    return_str = left + '+' + right
    del (left)
    del (right)
    return return_str

def strAbs(toAbs):
    # return_str = 'np.abs(' + toAbs + ')'
    # return_str = 'torch.abs(' + toAbs + ')'
    return_str = 'Abs(' + toAbs + ')'
    del(toAbs)

    return return_str


def isContainOperator(x):
    if x.find('+') > -1:
        return True
    if x.find('-') > -1:
        return True
    if x.find('*') > -1:
        return True
    if x.find('(') > -1:
        return True
    return False


def mul(left, right):
    left_str = left
    right_str = right

    if isContainOperator(left):
        left_str = '(' + left + ')'
        del(left)

    if isContainOperator(right):
        right_str = '(' + right + ')'
        del (right)

    return_str = left_str + '*' + right_str
    del (left_str)
    del (right_str)

    return return_str
def relu3(x, l, u):
    if l >= 0:
        return x
    if u <= 0:
        # return 0
        return str(0)
    if l < 0 and u >0:
        # return 0.5 * (x + sp.Abs(x))
        return_str = '0.5*(' + x + '+' + strAbs(x) + ')'
        del (x)
        return return_str



def relu3_deeppoly_low(x, l, u):
    if l >= 0:
        return x
    if u <= 0:
        # return 0
        return str(0)
    if l < 0 and u >0:
        # return 0
        return str(0)


def relu3_deeppoly_up(x, l, u):
    if l >= 0:
        return x
    if u <= 0:
        # return 0
        return str(0)
    if l < 0 and u >0:
        a = u/(u-l)

        if a > 0:
            a = str(a)
        else:
            a = "("+str(a)+")"

        b = - (u*l)/(u-l)

        if b > 0:
            b = str(b)
        else:
            b = "("+str(b)+")"

        c = a + "*(" + x+ ")+" + b
        return c
