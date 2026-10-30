# Detection evaluation assets

The COCO val2017 annotation subset in `coco_val2017_annotations.json` is
derived from COCO annotations, licensed under CC BY 4.0. Attribution:
Lin, Tsung-Yi, et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.

The ten source images are fetched from `images.cocodataset.org` at runtime.
Their COCO license IDs, license URLs, image URLs, and pinned SHA-256 values
are recorded in `coco_val2017_images.json`.

| COCO image | COCO license | License URL | Image URL |
| ---: | --- | --- | --- |
| 785 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000000785.jpg |
| 872 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000000872.jpg |
| 885 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000000885.jpg |
| 1000 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000001000.jpg |
| 1268 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000001268.jpg |
| 2473 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000002473.jpg |
| 5001 | Attribution License (ID 4) | http://creativecommons.org/licenses/by/2.0/ | http://images.cocodataset.org/val2017/000000005001.jpg |
| 274219 | No known copyright restrictions (ID 7) | http://flickr.com/commons/usage/ | http://images.cocodataset.org/val2017/000000274219.jpg |
| 427077 | No known copyright restrictions (ID 7) | http://flickr.com/commons/usage/ | http://images.cocodataset.org/val2017/000000427077.jpg |
| 560880 | No known copyright restrictions (ID 7) | http://flickr.com/commons/usage/ | http://images.cocodataset.org/val2017/000000560880.jpg |

The images are not distributed with AgentQuantix. The runtime verifies each
download against the SHA-256 and byte size in the manifest before use.
