#include "llvm/Support/raw_ostream.h"
#include <string>

#include "Param.h"

/*
    参数
*/
dacppTranslator::Param::Param() {
}

void dacppTranslator::Param::setRw(bool rw) {
    this->rw = rw;
}

bool dacppTranslator::Param::getRw() {
    return rw;
}

void dacppTranslator::Param::setType(std::string type) {
    const char *temp;
    const char *begin;
    const char *end;
    const char *comma;
    char *btype, *dim;
    int i;
    
    /* example: Tensor<int, 2>  */
    temp = type.c_str();
    begin = strchr (temp, '<');
    end = strrchr (temp, '>');
    comma = strrchr (temp, ',');

    /* NO error output, just check.  */
    if (strncmp (temp, "Tensor", 6) && 
        strncmp (temp, "const Tensor", strlen ("const Tensor"))) {
        this->basicType = "double";
        return;
    }

    assert (begin && end && begin < end);

    this->type = type;

    if (!comma || strchr(comma, '<') || strchr(comma, '(') ||
        strchr(comma, '{') ||
        (strchr(comma, '>') && end != strchr(comma, '>')) ||
        strchr(comma, ')') || strchr(comma, '}')) {
      /* Tensor has only 1 argument.  */
      btype = (char *)malloc(sizeof(char) * (end - begin));
      for (i = 1; begin + i < end; ++i)
        btype[i - 1] = begin[i];
      btype[i - 1] = '\0';
      this->basicType = btype;
      free(btype);
    } else {
      /* Tensor has 2 arguments.  */

      /* parse the first argument.  */
      btype = (char *)malloc(sizeof(char) * (comma - begin));
      for (i = 1; begin + i < comma; ++i)
        btype[i - 1] = begin[i];
      btype[i - 1] = '\0';
      this->basicType = btype;
      free(btype);

      /* parse the second argument.  */
      dim = (char *)malloc(sizeof(char) * (end - comma));
      for (i = 1; comma + i < end; ++i)
        dim[i - 1] = comma[i];
      dim[i - 1] = '\0';
      this->dim = atoi(dim);
      free(dim);
    }
}

std::string dacppTranslator::Param::getType() {
    return type;
}

std::string dacppTranslator::Param::getBasicType() {
    return basicType;
}

void dacppTranslator::Param::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Param::getName() {
    return name;
}

void dacppTranslator::Param::setShape(int size) {
    shape.push_back(size);
}

void dacppTranslator::Param::setShape(int idx, int size) {
    shape[idx] = size;
}

int dacppTranslator::Param::getShape(int idx) {
    return shape[idx];
}

int dacppTranslator::Param::getDim() {
    return shape.size(); 
}

/*
    划分结构参数
*/
dacppTranslator::ShellParam::ShellParam() {
}

void dacppTranslator::ShellParam::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::ShellParam::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::ShellParam::getNumSplit() {
    return splits.size();
}