# %%
from include import *
import sys

# %%
def main():
    pass

main_funcs = {
    "h": main_h,
    "rd": main_rd,
    "loadForPred": main_loadForPred,
    "loadForPred_rd": main_loadForPred_rd
}

if __name__ == "__main__":
    func_key = sys.argv[1] if len(sys.argv) > 1 else "h"
    main_func = main_funcs.get(func_key)
    if main_func:
        main_func()
    else:
        print("Function key not recognized.")


# %%
# if __name__ == "__main__":
#     # main_h()
#     # main_rd()
#     main_loadForPred()
#     # main_loadForPred_rd()
