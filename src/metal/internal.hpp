#pragma once

// Internal ObjC++ accessors shared across metal/*.mm TUs.
// Not part of the public C++ API.

#ifdef __OBJC__
#import <Metal/Metal.h>
id<MTLDevice> welvet_metal_device_handle(void);
id<MTLLibrary> welvet_metal_library_handle(void);
#endif
