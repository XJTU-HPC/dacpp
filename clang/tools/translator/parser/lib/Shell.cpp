#include <string>
#include <iostream>

#include "clang/AST/AST.h"

#include "Param.h"
#include "Shell.h"
#include "ASTParse.h"

struct symtab_symbol {
    int id;
    int cls;
    clang::ValueDecl *D;
    struct symtab* ste_table;
} ;

struct symtab {
    /* table of entries.  */
    symtab_symbol **symbols;

    /* number of actual entries entered in the table.  */
    int count;

    /* number of entries allocated currently.  */
    int allocated;

    struct symtab* parent;
    int next_id;
} ;

static void
ste_dealloc(symtab_symbol *ste)
{
    free (ste);
}

static void destroy_symtab (symtab_t* symtab)
{
    int i;
    if (!symtab) return;
    for (i = 0; i < symtab->count; ++i)
        ste_dealloc (symtab->symbols[i]);
    free (symtab->symbols);
    free (symtab);
}

static struct symtab* build_symtab (symtab_t* parent)
{
    symtab_t* symtab = NULL;

    symtab = (symtab_t*) malloc (sizeof (struct symtab));
    if (symtab == NULL)
        goto quit;

    memset (symtab, 0, sizeof (*symtab));
    symtab->parent = parent;

    return symtab;
quit:
    if (symtab) destroy_symtab (symtab);
    assert (0);
    return NULL;
}

static symtab_symbol *
ste_new(symtab_t *st)
{
    symtab_symbol *ste = NULL;
    symtab_t *parent = st;

    /* 分配内存。  */
    ste = (symtab_symbol *) malloc (sizeof (symtab_symbol));
    if (ste == NULL)
        goto fail;

    memset (ste, 0, sizeof (symtab_symbol));

    /* 插入符号表。  */
    ste->ste_table = st;
    if (st->allocated <= st->count)
    {
        st->symbols = (symtab_symbol **)realloc (st->symbols, (1 + st->allocated) * sizeof (symtab_symbol *));
        if (st->symbols == NULL)
            goto fail;
        st->allocated += 1;
    }
    st->symbols[st->count++] = ste;

    /* 更新符号表项标识符。  */
    while (parent->parent)
        parent = parent->parent;
    ste->id = ++parent->next_id;
    ste->cls = ste->id;

    return ste;
 fail:
    ste_dealloc (ste);
    assert (0);
    return NULL;
}

static symtab_symbol *search_symbol(symtab_t* symtab, const clang::ValueDecl *sym_name)
{
    int i;

    for (i = 0; i < symtab->count; ++i)
        if (sym_name == symtab->symbols[i]->D)
            return symtab->symbols[i];
    return symtab->parent ? search_symbol (symtab->parent, sym_name) : NULL;
}


/**
 * 存储划分结构信息类实现
 */
dacppTranslator::Shell::Shell() {
    this->symtab = build_symtab (NULL);
}

dacppTranslator::Shell::~Shell() {
    destroy_symtab (this->symtab);
}

void dacppTranslator::Shell::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Shell::getName() {
    return name;
}

void dacppTranslator::Shell::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Shell::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Shell::getNumParams() {
    return params.size();
}

void dacppTranslator::Shell::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::Shell::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::Shell::getNumSplits() {
    return splits.size();
}

void dacppTranslator::Shell::setShellParam(ShellParam* param) {
    shellParams.push_back(param);
}

dacppTranslator::ShellParam* dacppTranslator::Shell::getShellParam(int idx) {
    return shellParams[idx];
}

int dacppTranslator::Shell::getNumShellParams() {
    return shellParams.size();
}

void dacppTranslator::Shell::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Shell::getFather() {
    return father;
}

void dacppTranslator::Shell::setShellLoc(FunctionDecl* shellLoc) {
    this->shellLoc = shellLoc;
}

FunctionDecl* dacppTranslator::Shell::getShellLoc() {
    return shellLoc;
}

struct Visitor : DeclPrinter {
    ASTContext &Context;
    dacppTranslator::Shell *sh;
    PrinterHelper *helper;
    const PrintingPolicy &Policy;
    raw_ostream &os;
    const BinaryOperator* dacExpr;
  
    Visitor(dacppTranslator::Shell *sh, const BinaryOperator* dacExpr, raw_ostream &os, PrinterHelper *helper,
                const PrintingPolicy &Policy, unsigned Indentation = 0,
                StringRef NL = "\n", ASTContext *Context = nullptr)
        : DeclPrinter(os, helper, Policy, *Context, Indentation, NL), Context(*Context)
        , sh(sh), helper(helper), Policy(Policy), os(os), dacExpr(dacExpr) {}

    Expr *ignoreImplicitSemaNodes(Expr *Node) {
        Expr *E = Node;
        while (true) {
          if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
            E = MTE->getSubExpr();
          if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
            E = BTE->getSubExpr();
          if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
            E = ICE->getSubExpr();
          else
            break;
        }
    
        return E;
    }

    void VisitVarDecl(VarDecl *D) override;
    void VisitCallExpr(CallExpr *Call) override;
};

void Visitor::VisitCallExpr(CallExpr *Call) {
  std::string TempString;
  llvm::raw_string_ostream SS(TempString);
  unsigned i, e;
  symtab_symbol *vars[2];

  Call->getCallee()->printPretty(SS, nullptr, Policy);
  if (!strcmp(SS.str().c_str(), "binding")) {
    for (i = 0, e = Call->getNumArgs(); i != e; ++i) {
      if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }
      if (Expr *tempExpr = ignoreImplicitSemaNodes(Call->getArg(i)))
        if (CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(tempExpr))
          if (CCE->getNumArgs() == 1) {
            tempExpr = ignoreImplicitSemaNodes(CCE->getArg(0));
            if (const auto *DeclRef = dyn_cast<DeclRefExpr>(tempExpr))
              /* 应该总是能找到。  */
              vars[i] = search_symbol(sh->symtab, DeclRef->getDecl());
          }
    }
    vars[0]->cls = (vars[0]->cls > vars[1]->cls) ? vars[1]->cls : vars[0]->cls;
    vars[1]->cls = vars[0]->cls;
  }
}

void Visitor::VisitVarDecl (VarDecl *D)
{
  VarDecl *curVarDecl = D;
  do
  /*
      只解析两种节点
      1.算子(目前只有降维算子以及规则分区算子)
      2.dacpp::list
  */
  {
    if (curVarDecl->getType().getAsString().compare("dacpp::list") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::Index") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::RegularSplit") !=
            0) {
      break;
    }

    // 解析降维算子
    if (curVarDecl->getType().getAsString().compare("dacpp::Index") == 0) {
      dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit();
      sp->setId(curVarDecl->getNameAsString());
      sp->type = "IndexSplit";
      sh->setSplit(sp);
      sp->ste = ste_new(sh->symtab);
      sp->ste->D = curVarDecl;
      break;
    }

    // 解析规则分区算子
    if (curVarDecl->getType().getAsString().compare("dacpp::RegularSplit") ==
        0) {
      dacppTranslator::RegularSplit *sp = new dacppTranslator::RegularSplit();
      sp->setId(curVarDecl->getNameAsString());
      CXXConstructExpr *CCE =
          dacppTranslator::getNode<CXXConstructExpr>(curVarDecl->getInit());
      int count = 0;
      for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                          E = CCE->arg_end();
           I != E; ++I) {
        if (count == 1) {
          /* TODO: 计算常量表达式的值。  */
          sp->setSplitSize(std::stoi(
              (dyn_cast<IntegerLiteral>(*I))->getValue().toString(10, true)));
        } else if (count == 2) {
          /* TODO: 计算常量表达式的值。  */
          sp->setSplitStride(std::stoi(
              (dyn_cast<IntegerLiteral>(*I))->getValue().toString(10, true)));
        }
        count++;
      }
      sp->type = "RegularSplit";
      sh->setSplit(sp);
      sp->ste = ste_new (sh->symtab);
      sp->ste->D = curVarDecl;
      break;
    }

    /*
        解析dacpp::list
        对数据的所有划分信息均在dacpp::list的初始化列表中
    */
    InitListExpr *ILE =
        dacppTranslator::getNode<InitListExpr>(curVarDecl->getInit());
    for (unsigned int i = 0; i < ILE->getNumInits(); i++) {
      dacppTranslator::ShellParam *shellParam =
          new dacppTranslator::ShellParam();
      Expr *curExpr = ILE->getInit(i);
      std::vector<Expr *> astExprs;

      std::string name = "";
      dacppTranslator::getSplitExpr(curExpr, name, astExprs);
      shellParam->setName(name);
      for (unsigned int paramsCount = 0;
           paramsCount < sh->getShellLoc()->getNumParams(); paramsCount++) {
        if (shellParam->getName() !=
            sh->getShellLoc()->getParamDecl(paramsCount)->getNameAsString()) {
          continue;
        }
        shellParam->setRw(
            dacppTranslator::inputOrOutput(sh->getShellLoc()
                                               ->getParamDecl(paramsCount)
                                               ->getType()
                                               .getAsString()));
        shellParam->setType(sh->getParam(paramsCount)->getType());
        shellParam->setName(sh->getParam(paramsCount)->getName());
        for (int shapeIdx = 0; shapeIdx < sh->getParam(paramsCount)->getDim();
             shapeIdx++) {
          shellParam->setShape(sh->getParam(paramsCount)->getShape(shapeIdx));
        }
      }
      for (unsigned int i = 0; i < astExprs.size(); i++) {
        if (dacppTranslator::getNode<DeclRefExpr>(astExprs[i])) {
          VarDecl *vd = dyn_cast<VarDecl>(
              dacppTranslator::getNode<DeclRefExpr>(astExprs[i])->getDecl());

          if (vd->getType().getAsString().compare("dacpp::RegularSplit") == 0) {
            dacppTranslator::RegularSplit *sp =
                new dacppTranslator::RegularSplit();
            sp->type = "RegularSplit";
            sp->setId(dacppTranslator::getNode<StringLiteral>(vd->getInit())
                          ->getString()
                          .str());
            sp->setDimIdx(i);
            CXXConstructExpr *CCE =
                dacppTranslator::getNode<CXXConstructExpr>(vd->getInit());
            int count = 0;
            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                E = CCE->arg_end();
                 I != E; ++I) {
              if (count == 1) {
                /* TODO: 计算常量表达式的值。  */
                sp->setSplitSize(std::stoi((dyn_cast<IntegerLiteral>(*I))
                                               ->getValue()
                                               .toString(10, true)));
              } else if (count == 2) {
                /* TODO: 计算常量表达式的值。  */
                sp->setSplitStride(std::stoi((dyn_cast<IntegerLiteral>(*I))
                                                 ->getValue()
                                                 .toString(10, true)));
              }
              count++;
            }
            /* TODO: 处理除不尽的情况。  */
            sp->setSplitNumber((shellParam->getShape(i) - sp->getSplitSize()) /
                                   sp->getSplitStride() +
                               1);
            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit *isp =
                    static_cast<dacppTranslator::RegularSplit *>(
                        sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          } else if (vd->getType().getAsString().compare("dacpp::Index") == 0) {
            dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit();
            sp->type = "IndexSplit";
            sp->setId(dacppTranslator::getNode<StringLiteral>(vd->getInit())
                          ->getString()
                          .str());
            sp->setDimIdx(i);
            sp->setSplitNumber(shellParam->getShape(i));
            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit *isp =
                    static_cast<dacppTranslator::IndexSplit *>(sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          }
        } else {
          dacppTranslator::Split *sp = new dacppTranslator::Split();
          sp->type = "Split";
          sp->setId("void");
          sp->setDimIdx(i);
          shellParam->setSplit(sp);
        }
      }
      sh->setShellParam(shellParam);
    }
  } while (0);

  DeclPrinter::VisitVarDecl(D);
}

/*
 * GetBindInfo
 *
 * 目的：
 *  计算两算子是否属于同一集合；获取两算子偏移量。
 *
 * 参数：
 *  var1            算子1。
 *  var2            算子2。
 *  pbindInfo       两算子偏移量。
 *
 * 返回值：
 *  bool            两算子是否属于同一集合。 若属于，则通过pbindInfo返回偏移量。
 */

bool dacppTranslator::Shell::GetBindInfo(clang::ValueDecl *var1,
                                         clang::ValueDecl *var2,
                                         std::string *pbindInfo) {
  bool bResult;
  symtab_symbol *vars[2];

  vars[0] = search_symbol(symtab, var1);
  vars[1] = search_symbol(symtab, var2);
  bResult = vars[0] && vars[1] && vars[0]->cls == vars[1]->cls;
  if (pbindInfo && bResult) {
    pbindInfo->clear();
    pbindInfo->append("0");
  }
  return bResult;
}

// 解析Shell节点，将解析到的信息存储到Shell类中
void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
  Visitor V(this, dacExpr, SS, nullptr, PrintingPolicy(LangOptions()), 0, "\n",
            nullptr);

  /*
      数据关联计算表达式节点为一个BinaryOperator节点
      左值为CallExpr节点，表示shell函数调用
      右值为DeclRefExpr，表示calc函数引用
  */
  Expr *dacExprLHS = dacExpr->getLHS();
  CallExpr *shellCall = getNode<CallExpr>(dacExprLHS);
  FunctionDecl *shellFunc = shellCall->getDirectCallee();

  // 设置 AST 中 Shell 节点的位置
  setShellLoc(shellFunc);

  // 设置 Shell 函数名称
  setName(shellFunc->getNameAsString());

  // 设置 Shell 参数列表
  for (unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams();
       paramsCount++) {
    Param *param = new Param();

    // 获取参数读写属性
    param->setRw(inputOrOutput(
        shellFunc->getParamDecl(paramsCount)->getType().getAsString()));

    // 设置参数类型
    std::string type =
        shellFunc->getParamDecl(paramsCount)->getType().getAsString();
    param->setType(type);

    // 设置参数名称
    param->setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());

    // 设置参数形状
    for (unsigned int i = 0; i < shapes[paramsCount].size(); i++) {
      param->setShape(shapes[paramsCount][i]);
    }

    setParam(param);
    }

    // 获取shell函数体
    Stmt* shellFuncBody = shellFunc->getBody();
    V.Visit (shellFuncBody);
}
