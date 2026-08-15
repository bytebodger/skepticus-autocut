# LUTs

The grade is a single versioned 3D LUT, not a stack of `eq` parameters. Build it
once in DaVinci Resolve (free):

1. Grade a representative still from your studio setup.
2. Export as a `.cube` file.
3. Drop it here and reference it from the EDL's `grade` field.

**Version the filename.** When you change the look in six months you want to know
which episodes used which grade. Hence `skepticus_v1.cube`, `skepticus_v2.cube`, …

`skepticus_v1.identity.cube` in this folder is a no-op identity LUT so the grade
stage runs end-to-end before you've authored a real look. Replace it — do not
ship with it.
