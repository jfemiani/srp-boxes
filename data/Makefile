
# This may be set as an environment variable instead
VISION_FILE_SERVER ?= http://handouts1.csi.miamioh.edu:82/vision/

.PHONY: data volume-data annotoations imagery

help: _help

## Download all data from our server
data:  volume-data annotations imagery  

## Download the hand-done annotations from our server.
annotations: raw/srp/box-annotations.geojson 

## Download the high resolution aerial imagery from our server.
imagery: raw/srp/sec11-26949.tif 

## Download the volumetric density (derived from .laz files) from out server.
volume-data: interim/srp/stack_a4b2/lidar_volume.vrt 


## Delete our local copy of the aerial imagery. 
clean-imagery: 
	rm raw/srp/sec11-26949.tif

## Delete our local copy of the volume data. 
clean-volume-data: 
	rm -r interim/srp/stack_a4b2
	rm interim/srp/stack_a4b2.zip


# Undocumented targets
interim/srp/stack_a4b2.md5:
	mkdir -p interim/srp
	wget -O interim/srp/stack_a4b2.md5 $(VISION_FILE_SERVER)srp/stack_a4b2.md5

interim/srp/stack_a4b2.zip: interim/srp/stack_a4b2.md5
	# mkdir -p interim/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/stack_a4b2.zip
	wget -O $@.md5 $(VISION_FILE_SERVER)srp/stack_a4b2.zip.md5
	cd interim/srp && md5sum -c stack_a4b2.zip.md5
	# rm stack_a4b2.zip*


interim/srp/stack_a4b2/lidar_volume.vrt: interim/srp/stack_a4b2.zip
	unzip -o -d interim/srp interim/srp/stack_a4b2.zip
	cd interim/srp && md5sum -c stack_a4b2.md5


raw/srp/box-annotations.geojson:
	mkdir -p raw/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/box-annotations.geojson
	wget -O $@.md5 $(VISION_FILE_SERVER)srp/box-annotations.geojson.md5
	cd raw/srp && md5sum -c box-annotations.geojson.md5

raw/srp/sec11-26949.tif:
	mkdir -p raw/srp
	wget -O $@ $(VISION_FILE_SERVER)srp/sec11-26949.tif
	wget -O $@.md5 $(VISION_FILE_SERVER)srp/sec11-26949.tif.md5
	cd raw/srp && md5sum -c sec11-26949.tif.md5

# raw/srp/srp-test-site-1-laz.zip:
# 	mkdir -p raw/srp
# 	wget -O $@ $(VISION_FILE_SERVER)srp/srp-test-site-1-laz.zip
# 
# raw/srp/srp-test-site-1.laz: raw/srp-test-site-1.laz.zip
# 	unzip $^
# 
# tiles/64x64x6:
# 	python -m srp.data.train_test_split data/srp
# 


# A target for generating help messages   

# COLORS
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)


TARGET_MAX_CHAR_NUM=20
_help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
