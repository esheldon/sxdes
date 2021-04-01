# sxdes
Run the sep code with DES settings

Examples
---------
```python
import sxdes

# get the catalog and seg map using the convenience function
cat, seg = sxdes.run_sep(image, noise)

# using a mask
cat, seg = sxdes.run_sep(image, noise, mask=mask)
```

TODO
----

Figure out why `fluxerr_auto` is always returned by sep as zero
