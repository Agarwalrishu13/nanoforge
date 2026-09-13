# makes `tests` a regular package — some environments ship a stray `tests`
# package that would otherwise shadow this one (namespace packages lose to
# regular ones regardless of sys.path order)
