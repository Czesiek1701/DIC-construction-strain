# DIC-construction-strain
Project realised during **Smart Measurement Systems** course. Program is calculating a movement of points (based on their surrounding) and strain of structure. 

## DIC method

For each cell sums above are calulated. It main that for each nearest area of cell this expression is received and saved. Phase with minimum correlation value is selected and accepted as movement of the part of the image.

<img src=https://github.com/Czesiek1701/DIC-construction-strain/assets/157902583/f11728b9-72b7-44ee-b02d-48172930614a width="400">

Program has implemented a second alghoritm of DIC which hearth is shown below. It needs additional image processing as detecting edges and binarisation by quantille. This method is slower but in some cases and good parameteers chosing gives better result.

<img src=https://github.com/Czesiek1701/DIC-construction-strain/assets/157902583/6a7c4d3b-b4a4-422e-af69-56a86be51be7 width="400">

## Strain

Strain calculated as average of length and angle differentials on element as shown below.

<img src=https://github.com/Czesiek1701/DIC-construction-strain/assets/157902583/6346570d-eb0a-4a91-9ac2-a734a5b366d3 width="400">
<br>
<img src=https://github.com/Czesiek1701/DIC-construction-strain/assets/157902583/e8e82fe7-38f0-4cd0-becb-19868c462ce3 width="400">


## Reults movies

[Objects tracking](https://youtu.be/zrzBHMZAEoY)

[Point tracking - first method](https://youtu.be/cNXW701fKU4)

[Strain - first method](https://youtu.be/ym__KsELxck)

[Point tracking - second method](https://youtu.be/z-nL-ZFQ45M)

[Strain - second method](https://youtu.be/HkWQuloM_p4)


## To improve

 - move DIC operations to indices on matrix to avoid copying all parts of raster
 - add third dimension (angle) to DIC method
 - make reference frame as exponential moving average
 - improve file support
 - some refactorisations, add control script
