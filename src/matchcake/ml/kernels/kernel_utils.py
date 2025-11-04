from ...operations import CompRzRz


def mrot_zz_template(param0, param1, wires):
    CompRzRz([param0, param1], wires=wires)
