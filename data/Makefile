

VISION_FILE_SERVER = http://handouts1.csi.miamioh.edu:82/vision/

RAW_DATA_FILES = data/raw/srp/box-annotations.geojson
RAW_DATA_FILES += data/raw/srp/sec11-26949.tif

INTERIM_DATA_FILES = data/interim/srp/lidar_volime.vrt


data: $(RAW_DATA_FILE) $(INTERIM_DATA_FILES)


# Fetch the RAW data {{{

data/interim/srp/lidar_volume.vrt:
	wget -D data/interim/srp $(VISION_FILE_SERVER)srp/lidar_volume.zip
	unzip -d data/interim/srp data/interim/srp/lidar_volume.zip

data/raw/srp/box-annotations.geojson:
	wget -O $@ $(VISION_FILE_SERVER)srp/box-annotations.geojson

data/raw/srp/sec11-26949.tif:
	wget -O $@ $(VISION_FILE_SERVER)srp/sec11-26949.tif

# Targets that generate the VRT {{{

# These _should_ be part of the project, but our starting point is the volumetric 
# imagery for now because it took so long to generate the volume. We do not want
# to treat point clouds as source. 

data/raw/srp/srp-test-site-1-laz.zip:
	wget -O $@ $(VISION_FILE_SERVER)srp/srp-test-site-1-laz.zip

data/raw/srp/srp-test-site-1.laz: data/raw/srp-test-site-1.laz.zip
	unzip $^
#}}}


#}}}

tiles/64x64x6:
	python -m srp.data.train_test_split data/srp




