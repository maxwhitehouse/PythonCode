## IM USING MY ONE FREEBIE


import arcpy
import pandas as pd
import math
import matplotlib.pyplot as plt


class SmartRaster(arcpy.Raster):

    def __init__(self, raster_path):
        super().__init__(raster_path)
        self.raster_path = raster_path
        self.metadata = self._extract_metadata()

    def _extract_metadata(self):
        desc = arcpy.Describe(self.raster_path)
        extent = desc.extent
        
        bounds = [[extent.XMin, extent.YMax],
                  [extent.XMax, extent.YMin]]

        y_dim = self.height
        x_dim = self.width
        n_bands = self.bandCount
        pixelType = self.pixelType

        return {
            "bounds": bounds, 
            "x_dim": x_dim, 
            "y_dim": y_dim, 
            "n_bands": n_bands, 
            "pixelType": pixelType
        }

    def calculate_ndvi(self, band4_index=4, band3_index=3):
        """Calculate NDVI using the NIR and Red bands (typically Band 4 and Band 3)."""
        from arcpy import sa
        arcpy.CheckOutExtension("Spatial")

        okay = True
        ndvi = None
        try:
            # Build full paths to the bands inside the raster dataset
            nir_path = self.raster_path + f"/Band_{band4_index}"
            red_path = self.raster_path + f"/Band_{band3_index}"

            # Load NIR and Red bands
            nir = sa.Raster(nir_path)
            red = sa.Raster(red_path)

            # NDVI calculation: (NIR - Red) / (NIR + Red), avoiding divide-by-zero
            ndvi = sa.Con((nir + red) == 0, 0, (nir - red) / (nir + red))

            print("✅ NDVI calculated successfully.")

        except Exception as e:
            okay = False
            print(f"❌ Error calculating NDVI: {e}")
            ndvi = None

        return okay, ndvi








# Potential smart vector layer

# Potential smart vector layer
class SmartVectorLayer:
    def __init__(self, feature_class_path):
        """Initialize with a path to a vector feature class"""
        self.feature_class = feature_class_path
        
        # Check if it exists
        if not arcpy.Exists(self.feature_class):
            raise FileNotFoundError(f"{self.feature_class} does not exist.")

    def summarize_field(self, field):
        okay = True
        try: 
            existing_fields = [f.name for f in arcpy.ListFields(self.feature_class)]
            if field not in existing_fields:
                okay = False
                print(f"The field {field} is not in the list of possible fields")
                return False, None
        except Exception as e:
            print(f"Problem checking the fields: {e}")

        try: 
            with arcpy.da.SearchCursor(self.feature_class, [field]) as cursor:
                vals = [row[0] for row in cursor if row[0] is not None and not math.isnan(row[0])]
            mean = sum(vals) / len(vals)
            return okay, mean
        except Exception as e:
            print(f"Problem calculating mean: {e}")
            okay = False
            return okay, None

    def zonal_stats_to_field(self, raster, output_field):
        try:
            existing_fields = [f.name for f in arcpy.ListFields(self.feature_class)]
            if output_field not in existing_fields:
                arcpy.AddField_management(self.feature_class, output_field, "DOUBLE")

            zonal_table = "in_memory/zonal_stats"
            arcpy.sa.ZonalStatisticsAsTable(
                self.feature_class, "OBJECTID", raster, zonal_table, "DATA", "MEAN"
            )

            zonal_fields = [f.name for f in arcpy.ListFields(zonal_table)]
            print("Zonal table fields:", zonal_fields)

            join_field_zonal = None
            for field in ["OBJECTID", "OBJECTID_1", "FID"]:
                if field in zonal_fields:
                    join_field_zonal = field
                    break
            print(f"🧩 Using join field from zonal table: {join_field_zonal}")

            if join_field_zonal is None:
                raise ValueError("No suitable join field found in zonal table.")

            arcpy.JoinField_management(
                self.feature_class, "OBJECTID", zonal_table, join_field_zonal, ["MEAN"]
            )

            with arcpy.da.UpdateCursor(self.feature_class, ["MEAN", output_field]) as cursor:
                for row in cursor:
                    row[1] = row[0]
                    cursor.updateRow(row)

            arcpy.Delete_management(zonal_table)
            print(f"Zonal statistics successfully written to field '{output_field}'.")

        except Exception as e:
            print(f"Error performing zonal statistics: {e}")

    def save_as(self, output_feature_class):
        try:
            arcpy.CopyFeatures_management(self.feature_class, output_feature_class)
            print(f"Feature class saved as {output_feature_class}")
        except Exception as e:
            print(f"Error saving feature class: {e}")

    def extract_to_pandas_df(self, fields=None):
        okay = True

        if fields is None:
            fields = [f.name for f in arcpy.ListFields(self.feature_class) if f.type not in ('Geometry', 'OID')]
        else:
            true_fields = [f.name for f in arcpy.ListFields(self.feature_class) if f.type not in ('Geometry', 'OID')]
            disallowed = [user_f for user_f in fields if user_f not in true_fields]
            if len(disallowed) != 0:
                print("Fields given by user are not valid for this table")
                print(disallowed)
                okay = False
                return okay, None

        try:
            with arcpy.da.SearchCursor(self.feature_class, fields) as cursor:
                rows = [row for row in cursor]
        except Exception as e:
            print(f"Error extracting rows with SearchCursor: {e}")
            okay = False
            return okay, None

        try:
            df = pd.DataFrame(rows, columns=fields)
        except Exception as e:
            print(f"Error converting to Pandas DataFrame: {e}")
            okay = False
            return okay, None

        return okay, df


# Uncomment this when you get to the appropriate block in the scripts
#  file and re-load the functions

class smartPanda(pd.DataFrame):

    # This next bit is advanced -- don't worry about it unless you're 
    # curious.  It has to do with the pandas dataframe
    # being a complicated thing that could be created from a variety
    #   of types, and also that it creates a new dataframe
    #   when it does operations.  The use of @property is called
    #   a "decorator".  The _constructor(self) is a specific 
    #   expectation of Pandas when it does operations.  This just
    #   tells it that when it does an operation, make the new thing
    #   into a special smartPanda type, not an original dataframe. 

    @property
    def _constructor(self):
        return smartPanda
    
    # here, just set up a method to plot and to allow
    #   the user to define the min and max of the plot. 


    def scatterplot(self, x_field, y_field, title=None, 
                    x_min=None, x_max=None, 
                    y_min=None, y_max=None):
        """Make a scatterplot of two columns, with validation."""

        # Validate
        for field in [x_field, y_field]:
            if field not in self.columns:
                raise ValueError(f"Field '{field}' not found in DataFrame columns.")

        # filter the range
        df_to_plot = self
        if x_min is not None:
            df_to_plot = df_to_plot[df_to_plot[x_field] >= x_min]
        if x_max is not None:
            df_to_plot = df_to_plot[df_to_plot[x_field] <= x_max]
        if y_min is not None:
            df_to_plot = df_to_plot[df_to_plot[y_field] >= y_min]
        if y_max is not None:
            df_to_plot = df_to_plot[df_to_plot[y_field] <= y_max]



        # Proceed to plot
        plt.figure(figsize=(8,6))
        plt.scatter(df_to_plot[x_field], df_to_plot[y_field])
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.title(title if title else f"{y_field} vs {x_field}")
        plt.grid(True)
        plt.show()


    def mean_field(self, field):
        """Get mean of a field, ignoring NaN."""
        return self[field].mean(skipna=True)

    def save_scatterplot(self, x_field, y_field, outfile, title=None, 
                    x_min=None, x_max=None, 
                    y_min=None, y_max=None):
        """Make a scatterplot of two columns, with validation."""

        # Validate
        for field in [x_field, y_field]:
            if field not in self.columns:
                raise ValueError(f"Field '{field}' not found in DataFrame columns.")

        # filter the range
        df_to_plot = self
        if x_min is not None:
            df_to_plot = df_to_plot[df_to_plot[x_field] >= x_min]
        if x_max is not None:
            df_to_plot = df_to_plot[df_to_plot[x_field] <= x_max]
        if y_min is not None:
            df_to_plot = df_to_plot[df_to_plot[y_field] >= y_min]
        if y_max is not None:
            df_to_plot = df_to_plot[df_to_plot[y_field] <= y_max]



        # Proceed to plot
        plt.figure(figsize=(8,6))
        plt.scatter(df_to_plot[x_field], df_to_plot[y_field])
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.title(title if title else f"{y_field} vs {x_field}")
        plt.grid(True)
        plt.savefig(outfile)
        plt.close()

    def plot_from_file(self, csv_control_file_path):
        #  This reads the file at csv_control_file_path
        #   and uses it to make a plot, and then save
        #   it.  

        #  First, use the pandas functionality to read the
        #    .csv file.  The file should have two columns:
        #   param and value
        #  the param is the name of the item of interest, for 
        #    example "in_file", and the value is the value 
        #    of that param. 
        #  The required params are:
        #     param   -> value type
        #     --------------------
        #     x_field -> string
        #     y_field -> string
        #     outfile -> string path to graphics file output
        #  Optional:
        #     x_min -> numeric
        #     x_max -> numeric
        #     y_min -> numeric
        #     y_max -> numeric
             

        try: 
            params = pd.read_csv(csv_control_file_path)
        except Exception as e:
            print(f"Problem reading the {csv_control_file_path}")
            return False
        
        # Then we'll turn it into a dictionary
        #  To do this, we'll use a new functionality that you'll
        #  see in Python a lot -- "zip".   Look it up!  
        # Also, you can see I'm doing a trick with the 
        #  definition of the dictionary -- 
        
        try:
            param_dict ={k.strip(): v for k,v in zip(params['Param'], params['Value'])}

        except Exception as e:
            print(f"Problem setting up dictionary: {e}")

        # Then check that the required params are present
        # required params
        required_params = ["x_field", "y_field", "outfile"]

        #  Use a list comprehension and the .keys() to test 
        #   if the required ones are in the dictionary

        missing = [m for m in required_params if m not in param_dict.keys()]
        if missing:
            print("The param file needs to have these additional parameters")
            print(missing)
            return False


        #  Now add in "None" vals for the 
        #    optional params if the user does not set them


        optional_params = ["x_min", "x_max", "y_min", "y_max"]
        # go through the optional params, and if one is 
        #   is not in the param_dict, add it but give it the 
        #   value of "None" so the plotter won't set it. 

       
        for p in optional_params:
            if p not in param_dict.keys():
                param_dict[p] = None

        # Finally, do the plot! 
        try:
            self.save_scatterplot(param_dict['x_field'], 
                               param_dict['y_field'], 
                               param_dict['outfile'], 
                               x_min = param_dict['x_min'], 
                               x_max = param_dict['x_max'],
                               y_min = param_dict['y_min'],
                               y_max = param_dict['y_max'])
            print(f"wrote to {params}")
            return True   # report back success
        except Exception as e:
            print(f"Problem saving the scatterplot: {e}")