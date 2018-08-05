

VISION_FILE_SERVER = http://handouts1.csi.miamioh.edu:82/vision/


data: $(RAW_DATA_FILE)  volume-data

# Fetch the RAW data {{{

annotations: raw/srp/box-annotations.geojson

imagery: raw/srp/sec11-26949.tif

volume-data: interim/srp/stack_a4b2/lidar_volume.vrt


interim/srp/stack_a4b2.vrt.zip:
	mkdir -p interim/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/stack_a4b2.vrt.zip

interim/srp/stack_a4b2/lidar_volume.vrt: interim/srp/stack_a4b2.vrt.zip
	unzip -n -d interim/srp interim/srp/stack_a4b2.vrt.zip

raw/srp/box-annotations.geojson:
	mkdir -p raw/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/box-annotations.geojson

raw/srp/sec11-26949.tif:
	mkdir -p raw/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/sec11-26949.tif

# Targets that generate the VRT {{{

# These _should_ be part of the project, but our starting point is the volumetric 
# imagery for now because it took so long to generate the volume. We do not want
# to treat point clouds as source. 

raw/srp/srp-test-site-1-laz.zip:
	mkdir -p raw/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/srp-test-site-1-laz.zip

raw/srp/srp-test-site-1.laz: raw/srp-test-site-1.laz.zip
	unzip $^
#}}}


#}}}

tiles/64x64x6:
	python -m srp.data.train_test_split data/srp




