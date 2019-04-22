# sxdes
Run the sep code with DES settings

Examples
---------
```python
import sxdes

# get the catalog and seg map using the convenience function
cat, seg = sxdes.run_sep(image, noise)

# same but using the SepRunner class
runner = SepRunner(image, noise)

cat = runner.cat
seg = runner.seg
```

TODO
----

Figure out why fluxerr_auto is always returned by sep as zero
