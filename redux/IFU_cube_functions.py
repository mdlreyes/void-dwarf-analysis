import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from input_output_functions import *
from icecream import ic


def slice_cube2img(
    cube, output_dir="./", output_base_name="IFU_cube_img", ver=False, wavelength=None, SED=None
):
    """
    Slice a 3D cube and save each 2D slice in the third dimension as an image file (PNG).
    Parameters:
        cube (numpy.ndarray): The 3D cube to be sliced.
        output_dir (str): The directory where the image files will be saved (default: "./").
        output_base_name (str): The base name of the output image files (default: "IFU_cube_img").
        ver (bool): If True, print cube statistics for each slice (default: False).
        wavelength (float or None): The wavelength value (optional). If provided, an annotation
                                    with this value will be added to the top-left corner of each image.
        SED (numpy.ndarray or None): The 2D matrix for the SED (optional). If provided, it will be plotted
                                    as an additional panel below the main image.
    Returns:
        None
    """
    # Determine the dimensions of the cube
    z, y, x = cube.shape
    # Find the image dimensions (assuming the image is square)
    if x == y:
        img_size = x
    else:
        raise ValueError("The image dimensions are not square.")

    # Create the output_dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if ver:
        print("Slicing received cube...")
        print("Saving images to:\n", output_dir)
        print("Images naming follows:", output_base_name)

    # Iterate through the third dimension (wavelength) and save each 2D slice as an image
    for i in range(z):
        img_slice = cube[i, :-1, :-1]

        img_name = f"{output_base_name}_{i:05d}.png"
        img_path = os.path.join(output_dir, img_name)

        # Create the figure and plot the image
        vmin = 0
        vmax = 4e-10
        img_slice[img_slice < vmin] = vmin
        fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={"height_ratios": [2, 1]})

        # Plot the main image
        imgNorm = LogNorm(vmin=vmin, vmax=vmax)
        ax[0].imshow(img_slice, cmap="Spectral", norm=imgNorm)
        #ax[0].imshow(img_slice, cmap="Spectral", norm=LogNorm(), vmin=vmin, vmax=vmax)       

        # Add annotation for the wavelength value (if provided)
        if wavelength is not None:
            wavenumber = wavelength[i]
            angstrom_symbol = "\u00C5"
            lambda_symbol = "\u03BB"
            req_text = f"{lambda_symbol}={round(wavenumber, 2)} {angstrom_symbol}"
            variable_text = "$f_{\lambda}$ $(erg\ cm^{-2} Hz^{-1}$ " + angstrom_symbol + "$^{-1})$"
            ax[0].text(
                0.03,
                0.07,
                req_text,
                transform=ax[0].transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.7),
            )
            ax[0].text(
                0.03,
                0.98,
                variable_text,
                transform=ax[0].transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.7),
            )
        ax[0].axis("off")

        # Plot the SED in the second row (if provided)
        if SED is not None:
            ax[1].plot(SED[0], SED[1], color="blue")
            ax[1].plot(SED[0][i],SED[1][i],color="red",marker="o")
            ax[1].set_xlabel(lambda_symbol+" ("+angstrom_symbol+")")
            ax[1].set_ylabel(variable_text)
        plt.subplots_adjust(hspace=0)  # Remove space between subplots

        plt.savefig(img_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

def read_IFU_wavelength(file_path):
    data_dict = {}
    with open(file_path, "r") as wavelength_file:
        linehead = "empty"
        integ_type = True
        wavelength_file_data = dict()
        while linehead != "lmax (A)":
            newline = wavelength_file.readline()
            linehead = newline[0:12].strip()
            if len(linehead) == 0:
                continue  # Skip empty lines
            elif integ_type == True:
                newdata = int(newline[13:].strip())
                integ_type = False
            elif integ_type == False:
                newdata = float(newline[13:].strip())
            wavelength_file_data[linehead] = newdata
        # Read the Wavelength bins into a numpy array
        newline = wavelength_file.readline()
        numbers = []
        nlambda = wavelength_file_data["nlambda"]
        while len(numbers) < nlambda:
            newline = wavelength_file.readline()
            numbers.extend([float(num) for num in newline.split()])
        wavelength_file_data["wavelengths"] = numbers
    return wavelength_file_data


######################################################################################
######################################################################################
######################################################################################
######################################################################################
def main():
    import argparse

    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Slice a 3D cube and save each 2D slice as an image file (PNG)."
    )
    # Add command-line arguments for IFU_file, output_dir, and output_base_name
    parser.add_argument("IFU_file", type=str, help="Name of the input IFU file.")
    parser.add_argument(
        "-pth",
        "--IFU_file_path",
        type=str,
        default="./",
        help="Path to the input IFU file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_images",
        help="Output directory for saving the image files (default: './output_images').",
    )
    parser.add_argument(
        "--output_base_name",
        type=str,
        default="IFU_cube_img",
        help="Base name of the output image files (default: 'IFU_cube_img').",
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    # Read the cube from the specified IFU_file
    wavelengths_data = read_IFU_wavelength(
        #args.IFU_file_path + "info_" + args.IFU_file + ".txt"
        args.IFU_file_path+args.IFU_file+"_info.txt"
    )
    header, cube = read_grafic2npcube(
        args.IFU_file_path + args.IFU_file + "_icube.dat", cube_statistics=True
    )

    SED_values=[]
    for i in range(cube.shape[0]):
        img_slice = cube[i, :-1, :-1]
        SED_values.append(np.sum(img_slice))

    # Assuming you have the following vectors:
    wavelengths = np.array(wavelengths_data["wavelengths"])
    SED_values = np.array(SED_values)

    # Combine the vectors into a 2D matrix
    SED_graph = np.column_stack((wavelengths, SED_values))
    SED_graph = [wavelengths, SED_values]
    

    # Call the slice_cube2img function with the specified arguments
    slice_cube2img(
        cube,
        output_dir=args.output_dir,
        output_base_name=args.output_base_name,
        wavelength=wavelengths_data["wavelengths"],
        ver=True,
        SED=SED_graph
    )


if __name__ == "__main__":
    main()


######################################################################################
######################################################################################
######################################################################################
######################################################################################
