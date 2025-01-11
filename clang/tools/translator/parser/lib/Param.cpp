#include <string>
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/LLVM.h"
#include "Param.h"

using namespace llvm;

using namespace llvm;

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

// When printing a reference, the referenced type might also be a reference.
// If so, we want to skip that before printing the inner type.
static clang::QualType skipTopLevelReferences(clang::QualType T) {
  if (auto *Ref = T->getAs<clang::ReferenceType>())
    return skipTopLevelReferences(Ref->getPointeeTypeAsWritten());
  return T;
}

static clang::QualType GetBaseType(clang::QualType T) {
  // FIXME: This should be on the Type class!
  clang::QualType BaseType = T;
  while (!BaseType->isSpecifierType()) {
    if (const clang::PointerType *PTy = BaseType->getAs<clang::PointerType>())
      BaseType = PTy->getPointeeType();
    else if (const clang::ObjCObjectPointerType *OPT =
                 BaseType->getAs<clang::ObjCObjectPointerType>())
      BaseType = OPT->getPointeeType();
    else if (const clang::BlockPointerType *BPy = BaseType->getAs<clang::BlockPointerType>())
      BaseType = BPy->getPointeeType();
    else if (const clang::ArrayType *ATy = dyn_cast<clang::ArrayType>(BaseType))
      BaseType = ATy->getElementType();
    else if (const clang::FunctionType *FTy = BaseType->getAs<clang::FunctionType>())
      BaseType = FTy->getReturnType();
    else if (const clang::VectorType *VTy = BaseType->getAs<clang::VectorType>())
      BaseType = VTy->getElementType();
    else if (const clang::ReferenceType *RTy = BaseType->getAs<clang::ReferenceType>())
      BaseType = RTy->getPointeeType();
    else if (const clang::AutoType *ATy = BaseType->getAs<clang::AutoType>())
      BaseType = ATy->getDeducedType();
    else if (const clang::ParenType *PTy = BaseType->getAs<clang::ParenType>())
      BaseType = PTy->desugar();
    else
      // This must be a syntax error.
      break;
  }
  return BaseType;
}

void dacppTranslator::Param::setType(clang::QualType newType) {
  this->newType = newType;
  clang::SplitQualType split;
  split = newType.split();
  const clang::Type *ty = split.Ty;
  bool found_p = false;
  if (const clang::LValueReferenceType *LRT =
          dyn_cast<clang::LValueReferenceType>(ty)) {
    clang::QualType Inner =
        skipTopLevelReferences(LRT->getPointeeTypeAsWritten());
    clang::SplitQualType Split = Inner.split();
    if (const auto *ET = dyn_cast<clang::ElaboratedType>(Split.Ty)) {
      clang::NestedNameSpecifier *Qualifier = ET->getQualifier();
      if (!(Qualifier &&
            Qualifier->getKind() == clang::NestedNameSpecifier::Namespace &&
            strcmp(Qualifier->getAsNamespace()->getNameAsString().c_str(),
                   "dacpp"))) {
        if (!ET->getOwnedTagDecl()) {
          Split = ET->getNamedType().split();
          if (const auto *SpecTy =
                  dyn_cast<clang::TemplateSpecializationType>(Split.Ty)) {
            std::string BSBuf;
            llvm::raw_string_ostream BSStream(BSBuf);
            SpecTy->getTemplateName().print(
                BSStream, clang::LangOptions(),
                clang::TemplateName::Qualified::None);
            if (!strcmp("Tensor", BSBuf.c_str()) ||
                !strcmp("Vector", BSBuf.c_str()) ||
                !strcmp("Matrix", BSBuf.c_str())) {
              found_p = true;
              llvm::ArrayRef<clang::TemplateArgument> Args =
                  SpecTy->template_arguments();
              bool __attribute__((unused)) FirstArg = true;
              for (const auto &Arg : Args) {
                // Print the argument into a string.
                llvm::SmallString<128> Buf;
                llvm::raw_svector_ostream ArgOS(Buf);
                const clang::TemplateArgument &Argument = (Arg);
                if (Argument.getKind() == clang::TemplateArgument::Pack) {
                } else {
                  if (clang::TemplateArgument::Type == Argument.getKind()) {
                    newType = Argument.getAsType();
                    break;
                  }
                }
                if (!FirstArg)
                  llvm_unreachable("unreachable");
                FirstArg = false;
              }
            }
          }
        }
      }
    }
  }
  if (!found_p)
    newType = GetBaseType(newType);
  this->BasicType = newType;
}

std::string dacppTranslator::Param::getType() {
    return newType.getAsString();
}

std::string dacppTranslator::Param::getBasicType() {
    return BasicType.getAsString();
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