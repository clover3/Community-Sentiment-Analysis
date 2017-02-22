package sfc
/**
  * Created by user on 2017-02-22.
  */
package object tag {
  type WordID = Int
  type SynsetIdx = Int
  type SynsetOffset = Int
  type StdIdx = Int

  class Tag(id: StdIdx, parentID: List[SynsetOffset], val name: String) {
    def canEqual(a: Any) = a.isInstanceOf[Tag]
    override def equals(that: Any): Boolean = that match {
        case that: Tag => that.canEqual(this) && this.hashCode == that.hashCode
        case _ => false
      }
    override def hashCode: Int = {
      val prime = 31
      var result = 1
      result = prime * result + id;
      result = prime * result + (if (name == null) 0 else name.hashCode)
      return result
    }
  }

  class TagFinder(tags: Set[Tag]) {
    val nameIndex = (tags map (x => (x.name, x))).toMap

    def findByName(tagName: String): Tag = nameIndex(tagName)
    def findByNameEx(tagName: String): Tag = {
      val matches = tags filter (x => x.name.contains(tagName))
      matches foreach (x => println(x.name))
      matches.head
    }
  }

}