package sfc
import sfc._
import sfc.tag._
import sfc.sfc2._

/**
  * Created by user on 2017-02-22.
  */
package object list {

  class Generator(allTags: Set[Tag]) {
    val tagFinder = new TagFinder(allTags)

    val TagHumanLike : String = "person 0,individual 0,someone 0,somebody 0,mortal 0,human 0,soul 0 어떤__사람|156508_인간|"
    val TagVehicle : String = "vehicle 0 운송_수단/"
    val TagEntity : String = "entity 0 "
    val TagObject : String = "object 0,physical_object 0 "

    def ride = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      val vehicleTag: Tag = tagFinder.findByName(TagVehicle)
      val vehicle: Argument = new Argument(vehicleTag, "Target")
      val ride: SubcategorizationFrame = new SubcategorizationFrame("타다", List(actor, vehicle))
      ride
    }
    def see = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      // FIXME target can be any noun
      val targetTag: Tag = tagFinder.findByName(TagEntity)
      val target: Argument = new Argument(targetTag, "Target")
      val scf: SubcategorizationFrame = new SubcategorizationFrame("보다", List(actor, target))
      scf
    }
    def beingpretty = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      // FIXME target can be any noun
      val targetTag: Tag = tagFinder.findByName(TagEntity)
      val target: Argument = new Argument(targetTag, "Target")
      val scf: SubcategorizationFrame = new SubcategorizationFrame("이쁘다", List(actor, target))
      scf
    }
    def beingpretty2 = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      // FIXME target can be any noun
      val targetTag: Tag = tagFinder.findByName(TagEntity)
      val target: Argument = new Argument(targetTag, "Target")
      val scf: SubcategorizationFrame = new SubcategorizationFrame("예쁘다", List(actor, target))
      scf
    }

    def buy = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      // FIXME target can be any noun
      val targetTag: Tag = tagFinder.findByName(TagObject)
      val target: Argument = new Argument(targetTag, "Target")
      val scf: SubcategorizationFrame = new SubcategorizationFrame("살다", List(actor, target))
      scf
    }

    def buy2 = {
      val personTag: Tag = tagFinder.findByName(TagHumanLike)
      val actor: Argument = new Argument(personTag, "Actor")

      // FIXME target can be any noun
      val targetTag: Tag = tagFinder.findByName(TagObject)
      val target: Argument = new Argument(targetTag, "Target")
      val scf: SubcategorizationFrame = new SubcategorizationFrame("사다", List(actor, target))
      scf
    }
  }

}