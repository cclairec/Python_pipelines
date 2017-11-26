#!/bin/sh

umap_nii=$1         # Path to subject's umap
ute_folder=$2       # Path to subject's UTE umap dicom folder
subject_basename=$3

subject_directory=$(dirname ${umap_nii})

echo ${subject_basename}

seg_maths=`which seg_maths`
dcmdump=`which dcmdump`
dcmodify=`which dcmodify`

# Copy the original UTE umap dicom files
echo "Copy the original UTE umap dicom files"

mkdir UTE_umap_tmp

cp ${ute_folder}/* UTE_umap_tmp

# Rename the original UTE umap in dicom so that the ordering of the file names corresponds to their instance number
echo "Rename the original UTE umap"
for file in UTE_umap_tmp/*.dcm
do
    file_name=$(basename "$file" .dcm)

    IN=$(${dcmdump} $file +P 0020,0013)
    IN=${IN#*[}
    IN=${IN%]*}

    if [ $IN -lt 10 ]; then
        mv $file UTE_umap_tmp/"00"$IN"_"$file_name.dcm
    else if [ $IN -lt 100 ]; then
        mv $file UTE_umap_tmp/"0"$IN"_"$file_name.dcm
    else
        mv $file UTE_umap_tmp/$IN"_"$file_name.dcm
    fi fi
done


# Flip the umap
echo "Flip the umap"
fslswapdim ${umap_nii} x -y z ${subject_basename}_flipped.nii.gz

# From 3D to 2D slices
echo "3D to 2D slices"
mkdir ${subject_basename}_raw/
fslslice ${subject_basename}_flipped.nii.gz ${subject_basename}_raw/${subject_basename}

# Convert the 2D slices from nifti to nifti_pair format
echo "nifti to nifti_pair"
for file in ${subject_basename}_raw/*.nii.gz
do
    fslchfiletype NIFTI_PAIR $file
done

# Change the pixel datatype to unsigned short
echo "unsigned short"
for file in ${subject_basename}_raw/*.hdr
do
    ${seg_maths} $file -thr 0 -odt ushort $file
done


# Copy the renamed original UTE umap dicom files to the new umap folder
mkdir ${subject_basename}/
cp UTE_umap_tmp/* ${subject_basename}/

# Store raw binary files in a table
i=0
for file in ${subject_basename}_raw/*.img
do
    filename[$i]=$(basename "$file")
    filename[$i]="${filename[$i]%.*}"
    raws[$i]=$file
    i=$((i+1))
done

# Store original dicom files in a table
i=0
for file in ${subject_basename}/*.dcm
do
    dicoms[$i]=$file
    i=$((i+1))
done

# Rename original dicom files by raw binary files' name
count=${#dicoms[@]}
for i in `seq 1 $count`
do
    mv ${dicoms[$i-1]} ${subject_basename}/${filename[$i-1]}.dcm
done


# Store renamed dicom files in a table
i=0
for file in ${subject_basename}/*.dcm
do
    dicoms[$i]=$file
    i=$((i+1))
done

# Replace the pixel data of the renamed dicom files by the raw binary files
count=${#dicoms[@]}
for i in `seq 1 $count`
do
    ${dcmodify} -if "PixelData=${raws[$i-1]}" ${dicoms[$i-1]}
    ${dcmodify} -nb -m "0008,103e"="Head_MRAC_PET_${subject_basename}" ${dicoms[$i-1]}
done

# Zip the new umap folder to output directory
rm ${subject_basename}/*.dcm.bak
zip -j -r ${subject_basename}_DICOM.zip ${subject_basename}/*.dcm

# Delete intermediate files
rm -r ${subject_basename}_raw/
rm -r UTE_umap_tmp/
rm ${subject_basename}_flipped.nii.gz
