#ifndef DACPP_H
#define DACPP_H

#include "Slice.h"
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

#ifndef DACPP_IO
#define DACPP_IO(type) __attribute__((annotate(#type)))
#endif

#ifndef DACPP_REDUCE
#define DACPP_REDUCE(rule) __attribute__((annotate(#rule)))
#endif

#endif