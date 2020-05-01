from lib.tools import *

class program():
  def __init__(self,args):
    self.tool = tools(args)
    self.firstName = "firstName"
    self.lastName = "lastName"
    self.age = "age"
    self.setup(args)

  def setup(self,args):
    if self.tool.argHasValue("-f"):
      val=self.tool.argValue("-f")
      self.firstName=val

    if self.tool.argHasValue("-l"):
      val=self.tool.argValue("-l")
      self.lastName=val

    if self.tool.argHasValue("-a"):
      val=self.tool.argValue("-a")
      self.age=val

  def aff(self):
    print("{} {} : {}".format(self.firstName,self.lastName,self.age))

  def run(self):
    print("Current settings for {} :".format(self.firstName))
    self.aff()

    if self.tool.argExist("-h"):
      print("hello",self.firstName)

    print("Running !")