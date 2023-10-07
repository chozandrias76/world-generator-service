import cv2
from rivamap.rivamap import (
    preprocess,
    singularity_index,
    delineate,
    georef,
    visualization,
)


class River:
    def __init__(self, band_three_path, band_six_path):
        # "gis-raw/LC08_L1TP_014035_20231003_20231003_02_RT_B3.TIF"
        self.b3 = cv2.imread(band_three_path, cv2.IMREAD_UNCHANGED)
        # "gis-raw/LC08_L1TP_014035_20231003_20231003_02_RT_B6.TIF"
        self.b6 = cv2.imread(band_six_path, cv2.IMREAD_UNCHANGED)

    def compute(self):
        I1 = preprocess.mndwi(self.b3, self.b6)
        I2 = None
        print("Compute the modified normalized difference water index")
        cv2.imwrite(
            "gis-converted/LC08_L1TP_014035_20231003_20231003_02_RT_mndwi.TIF",
            cv2.normalize(I1, I2, 0, 255, cv2.NORM_MINMAX),
        )
        print("Create the filters that are needed to compute the multiscale singularity index")
        filters = singularity_index.SingularityIndexFilters()
        print("Apply the index to extract curvilinear structures from the input image")
        psi, widthMap, orient = singularity_index.applyMMSI(I1, filters)
        cv2.imwrite(
            "gis-converted/LC08_L1TP_014035_20231003_20231003_02_RT_psi.TIF",
            cv2.normalize(psi, None, 0, 255, cv2.NORM_MINMAX),
        )
        print("Extract and threshold centerlines to delineate rivers")
        nms = delineate.extractCenterlines(orient, psi)
        centerlines = delineate.thresholdCenterlines(nms)
        cv2.imwrite(
            "gis-converted/LC08_L1TP_014035_20231003_20231003_02_RT_nms.TIF",
            cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX),
        )
        print("Generate a map of the extracted channels")
        raster = visualization.generateRasterMap(centerlines, orient, widthMap)
        cv2.imwrite(
            "gis-converted/LC08_L1TP_014035_20231003_20231003_02_RT_rastermap.TIF",
            cv2.normalize(raster, None, 0, 255, cv2.NORM_MINMAX),
        )
        visualization.generateVectorMap(
            centerlines,
            orient,
            widthMap,
            saveDest="vis/LC08_L1TP_014035_20231003_20231003_02_RT_vector.pdf",
        )
        print("Create a quiver plot showing the magnitude and direction of channels")
        visualization.quiverPlot(
            psi,
            orient,
            saveDest="vis/LC08_L1TP_014035_20231003_20231003_02_RT_quiver.pdf",
        )
        print("Save the results as georeferenced files")
        gm = georef.loadGeoMetadata(
            "gis-raw/LC08_L1TP_014035_20231003_20231003_02_RT_B3.TIF"
        )
        psi = preprocess.contrastStretch(raster)
        psi = preprocess.double2im(raster, "uint16")
        georef.saveAsGeoTiff(
            gm,
            raster,
            "gis-converted/LC08_L1TP_014035_20231003_20231003_02_RT_raster_geotagged.TIF",
        )

        # Export the (coordinate, width) pairs to a comma separated text file
        georef.exportCSVfile(
            centerlines,
            widthMap,
            gm,
            "vis/LC08_L1TP_014035_20231003_20231003_02_RT_results.csv",
        )
        # Export line segments to a shapefile
        georef.exportShapeFile(
            centerlines,
            widthMap,
            gm,
            "vis/LC08_L1TP_014035_20231003_20231003_02_RT_results",
        )
