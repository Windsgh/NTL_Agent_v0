from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class NightlightDataInput(BaseModel):
    study_area: str = Field(..., description="Name of the study area of interest. Example:'南京市'")
    scale_level: str = Field(..., description="Scale level, e.g.'country', 'province', 'city', 'county'.")
    dataset_choice: str = Field(...,
                                description="Data type options: 'annual', 'monthly' or 'daily'.")
    time_range_input: str = Field(...,
                                  description="Time range in the format 'YYYY-MM to YYYY-MM'. Example: '2020-01 to 2020-02'")
    export_folder: str = Field(...,
                               description="The local folder path of the exported file. Example:'C:/NTL_Agent/Night_data/Nanjing'")
    collection_name: str = Field(None, description="The name of the collection")


def fetch_and_download_nightlight_data(
        study_area: str,
        scale_level: str,
        dataset_choice: str,
        time_range_input: str,
        export_folder: str,
):
    import re
    import os
    import ee
    import geemap
    import calendar
    from datetime import datetime, timedelta

    # Set administrative boundary dataset based on scale level
    national_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/World_countries")
    province_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/province")
    city_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/city")
    county_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/county")

    # Select administrative boundaries
    def get_administrative_boundaries(scale_level):
        # Handle directly governed cities as province-level data in China
        directly_governed_cities = ['北京市', '天津市', '上海市', '重庆市']
        if scale_level == 'province' or (scale_level == 'city' and study_area in directly_governed_cities):
            admin_boundary = province_collection
            name_property = 'name'
        elif scale_level == 'country':
            admin_boundary = national_collection
            name_property = 'NAME'
        elif scale_level == 'city':
            admin_boundary = city_collection
            name_property = 'name'
        elif scale_level == 'county':
            admin_boundary = county_collection
            name_property = 'name'
        else:
            raise ValueError("Unknown scale level. Options are 'country', 'province', 'city', or 'county'.")
        return admin_boundary, name_property

    admin_boundary, name_property = get_administrative_boundaries(scale_level)
    region = admin_boundary.filter(ee.Filter.eq(name_property, study_area))

    # Validate region
    if region.size().getInfo() == 0:
        raise ValueError(f"No area named '{study_area}' found under scale level '{scale_level}'.")

    # Parse time range
    def parse_time_range(time_range_input, dataset_choice):
        time_range_input = time_range_input.replace(' ', '')
        if 'to' in time_range_input:
            start_str, end_str = time_range_input.split('to')
            start_str, end_str = start_str.strip(), end_str.strip()
        else:
            # Single date input
            start_str = end_str = time_range_input.strip()

        if dataset_choice.lower() == 'annual':
            if not re.match(r'^\d{4}$', start_str) or not re.match(r'^\d{4}$', end_str):
                raise ValueError("Invalid annual format. Use 'YYYY' or 'YYYY to YYYY'.")
            start_date, end_date = f"{start_str}-01-01", f"{end_str}-12-31"
        elif dataset_choice.lower() == 'monthly':
            if not re.match(r'^\d{4}-\d{2}$', start_str) or not re.match(r'^\d{4}-\d{2}$', end_str):
                raise ValueError("Invalid monthly format. Use 'YYYY-MM' or 'YYYY-MM to YYYY-MM'.")
            start_year, start_month = map(int, start_str.split('-'))
            end_year, end_month = map(int, end_str.split('-'))
            start_date = f"{start_year}-{start_month:02d}-01"
            end_date = f"{end_year}-{end_month:02d}-{calendar.monthrange(end_year, end_month)[1]}"
        elif dataset_choice.lower() == 'daily':
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', start_str) or not re.match(r'^\d{4}-\d{2}-\d{2}$', end_str):
                raise ValueError("Invalid daily format. Use 'YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD'.")
            start_date, end_date = start_str, end_str
        else:
            raise ValueError("Invalid dataset choice.")

        if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
            raise ValueError("Start date cannot be later than end date.")

        return start_date, end_date

    start_date, end_date = parse_time_range(time_range_input, dataset_choice)
    NTL_type = "VIIRS"
    # Load image collection based on dataset choice
    if dataset_choice.lower() == 'annual':
        viirs_images = []
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
            if 2000 <= year <= 2023:
                collection = ee.ImageCollection('projects/sat-io/open-datasets/npp-viirs-ntl').filterDate(f'{year}-01-01',
                                                                                        f'{year + 1}-01-01')
                band = 'b1'
            else:
                return "Year out of data range, must be between 2000 and 2023"

            # 裁剪、均值计算和时间属性设置
            image = collection.select(band).filterBounds(region.geometry()).map(lambda img: img.clip(region)).mean()
            image = image.set('system:time_start', ee.Date(f'{year}-01-01').millis())
            viirs_images.append(image)

        # 检查影像集合是否为空
        if not viirs_images:
            return "No images found for the specified date range and region."

        NTL_collection = ee.ImageCollection(viirs_images)




    elif dataset_choice.lower() == 'monthly':

        start_year, start_month = map(int, start_date[:7].split('-'))

        end_year, end_month = map(int, end_date[:7].split('-'))

        viirs_images = []

        for year in range(start_year, end_year + 1):

            months = range(1, 13)

            if year < 2014:
                return "Date out of data range, Date should later than January 2014 "

            if year == start_year:
                months = range(start_month, 13)

            if year == end_year:
                months = range(1, end_month + 1)

            for month in months:

                if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                    continue

                start_day = f'{year}-{month:02d}-01'

                end_day = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]}'

                collection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate(start_day, end_day)

                image = collection.select('avg_rad').filterBounds(region.geometry()).map(lambda img: img.clip(region)).mean()

                image = image.set('system:time_start', ee.Date(start_day).millis())

                viirs_images.append(image)

        NTL_collection = ee.ImageCollection(viirs_images)





    elif dataset_choice.lower() == 'daily':
        start_year = int(start_date[:4])
        if start_year < 2014:
            return "Date out of data range, Date should later than January 2014 "
        NTL_collection = (
            ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
            .filterDate(start_date, (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
            .select('DNB_BRDF_Corrected_NTL')
            .filterBounds(region.geometry())
            .map(lambda image: image.clip(region))
        )
    else:
        raise ValueError("Invalid dataset choice.")

    # Export images to local directory
    os.makedirs(export_folder, exist_ok=True)
    images_list = NTL_collection.toList(NTL_collection.size())
    num_images = NTL_collection.size().getInfo()

    exported_files = []  # List to store file paths
    for i in range(num_images):
        image = ee.Image(images_list.get(i))
        if dataset_choice.lower() == 'daily':
            image_date = image.date().format('YYYY-MM-dd').getInfo()
        elif dataset_choice.lower() == 'monthly':
            image_date = image.date().format('YYYY-MM').getInfo()
        else:
            image_date = image.date().format('YYYY').getInfo()
        export_path = os.path.join(export_folder, f"NTL_{study_area}_{NTL_type}_{image_date}.tif")
        geemap.ee_export_image(
            ee_object=image,
            filename=export_path,
            scale=500,
            region=region.geometry(),
            crs='EPSG:4326',
            file_per_band=False
        )
        exported_files.append(export_path)  # Store file path
        print(f"Image exported to: {export_path}")
    return f"Data has been saved to the following locations: {', '.join(exported_files)}"


# Update the nightlight_download_tool
NTL_download_tool = StructuredTool.from_function(
    fetch_and_download_nightlight_data,
    name="NTL_download_tool",
    description=(
        """
        This tool download nighttime light data from Google Earth Engine based on specified parameters, 
        including region name, scale level('country', 'province', 'city', 'county'), data type('annual', 'monthly' or 'daily'), and time range('YYYY-MM to YYYY-MM'). 
        Regional names in China should be in Chinese (e.g., 江苏省, 南京市, 鼓楼区). 

        Example Input:
        NightlightDataInput(
            study_area='南京市',
            scale_level='city',
            dataset_choice='daily',
            time_range_input='2020-01-01 to 2020-02-01',
            export_folder='C:/NTL_Agent/Night_data/Nanjing',
        )
        or
        (
            study_area='China',
            scale_level='country',
            dataset_choice='annual',
            time_range_input='2020 to 2020',
        )
        or
        (
            study_area='黄浦区',
            scale_level='county',
            dataset_choice='daily',
            time_range_input='2020-01-01 to 2020-02-01',
        )
        """
    ),
    input_type=NightlightDataInput,
)
