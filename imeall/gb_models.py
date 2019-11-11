"""
This module contains key grain boundary data models for the
SQL database. Also operates as a command line tool to populate
local versions of the database.
"""
import os
import sys
import json
import glob
import logging
import argparse
import numpy as np

from peewee   import *
from datetime import datetime, timedelta
from models   import GBAnalysis, PotentialParameters
from quippy   import Atoms, set_fortran_indexing, Potential, AtomsReader
from imeall import app

set_fortran_indexing(False)
GRAIN_DATABASE = app.config['GRAIN_DATABASE']
database = SqliteDatabase(app.config['GRAIN_DATABASE_SQL'])

class BaseModel(Model):
    class Meta():
        database = database

class GrainBoundary(BaseModel):
    """GrainBoundary is a peewee :peewee:`Model` of the canonical parent grain.
    Model is connected to the SQL database of the grain boundaries. All vectors
    are serialized to csv string for storage.

    Attributes:
      orientation_axis(str): orientation axis defining the grain.
      angle(float): misorientation angle defining the grain (radians).
      boundary_plane(str): boundary plane normal defining the grain.
      path(str): Full directory string relative to the grain boundary database root.
      gb_type(str): Grain boundary type (tilt, twist, general).
      z_planes(list): Location of center of interfacial planes.
      n_at(int): Number of grains in parent.
    """
    orientation_axis = CharField()
    angle            = FloatField()
    boundary_plane   = CharField()
    path             = CharField()
    gb_type          = CharField()
    z_planes         = CharField()
    n_at             = IntegerField()
    coincident_sites = IntegerField()
    sigma_csl        = IntegerField(default=0)
    height           = FloatField()
    area             = FloatField()
    notes            = TextField(default="")
    gbid             = CharField(unique=True)

class SubGrainBoundary(BaseModel):
    """
    SubGrainBoundary Model.

    Attributes:
      path(str): relative to the `GRAIN_BOUNDARY` database root 
                 (set in the application instance/config.py file).
      canonical_grain(:class:GrainBoundary): Each :class:`SubGrainBoundary`  
      has a parent :class:`GrainBoundary` model.
      rbt(str): rigid body translations.
      potential(str): Potential parameter file.
      rcut(float): atom deletion criterion.
      area(float): grain boundary interfacial area.
      n_at(int): number of atoms in subgrain structure.
      E_gb(float): grain boundary energies.
      E_gb_init(float): Energy of initial unrelaxed structure.
      notes(str): Special notes of interest about this boundary.
      gbid(str): SubGrainBoundary id.

    """
    path            = CharField()
    canonical_grain = ForeignKeyField(GrainBoundary, "subgrains")
    converged       = BooleanField()
    rbt             = CharField()
    potential       = CharField()
    rcut            = FloatField()
    area            = FloatField()
    n_at            = IntegerField()
    E_gb            = FloatField(default=0.0)
    E_gb_init       = FloatField(default=0.0)
    notes           = TextField(default="")
    gbid            = CharField()
    #create integrity index so that we can't have two entries for same potential and gbid
    class Meta:
        indexes=(
                          (('potential', 'gbid'), True), #trailing comma is necessary
                        )

class Fracture(BaseModel):
    """Fracture data :py:class:`Model`.

    Attributes:
      fracture_system(str): Serialized string in form [crack direction](crack plane).
      G(float): stress energy release rate.
      strain_rate(float): time incremental rate of strain increase fs^{-1}
      sim_T(float): simulation temperature.
    """
    fracture_system = CharField()
    G               = FloatField()
    strain_rate     = FloatField()
    sim_T           = FloatField()
    notes           = CharField(default="")

class Dislocation(BaseModel):
    gbid  =  CharField()

def serialize_vector(vector):
    return ','.join(map(str, vector))

def deserialize_vector_float(ser_vec):
    return map(float, ser_vec.split(','))

def deserialize_vector_int(ser_vec):
    return map(int, ser_vec.split(','))

def create_tables(database):
    """ Create GrainBoundary, and SubGrainBoundary Tables.

    Args:
      database(:py:class:SqliteDatabase)
    """
    database.connect()
    database.create_tables([GrainBoundary,SubGrainBoundary], True)

def add_conv_key(material='alphaFe', or_axis='001'):
    """Check if subgb.json dictionary contains a convergence
    key. If not add key to subgb.json file and default to false.

    Args:
      material(str): name of material to search.
      or_axis(str): orientation axis.
    """

    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE,
                                      os.path.join(material, or_axis))), gb_files, 'gb.json')
    for gb in gb_files:
        print gb[0], gb[1]
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        for subgb_model in GB_model.subgrains:
            subgb_dict_path = os.path.join(subgb_model.path,'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            with open(subgb_dict_path,'r') as f:
                subgb_dict = json.load(f)
            try:
                print subgb_dict['gbid'], subgb_dict['converged']
            except KeyError:
                print 'Adding Convergence Keys'
                subgb_dict['converged'] = False
                with open(subgb_dict_path,'w') as f:
                    json.dump(subgb_dict, f, indent=2)

def gb_check_dir_integrity(material='alphaFe', or_axis='001'):
    """Check if directory in directory tree contains
    a grain boundary json file, and update sql database to include the
    parent grain if it is missing.
    """
    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE,
                                      os.path.join(material, or_axis))), gb_files, 'gb.json')
    for gb in gb_files:
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        for subgb_model in GB_model.subgrains:
            subgb_dict_path = os.path.join(subgb_model.path,'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            try:
                with open(subgb_dict_path,'r') as f:
                    subgb_dict = json.load(f)
            except IOError:
                NOINPUT = True
                print subgb_dict_path
                while NOINPUT:
                    user_input = raw_input("Directory missing delete model (y/n)?")
                    if user_input == 'y':
                        print 'Deleting Model'
                        subgb_model.delete_instance()
                        NOINPUT=False
                    elif user_input =='n':
                        print 'Keeping Model'
                        NOINPUT=False
                    else:
                        pass

def gb_check_path(material='alphaFe', or_axis='001', modify_db=False):
    """Compare consistency between location of subgb.json
    paths and the paths in the closure tree. If path is missing
    from directory tree delete from the SQL database.

    Args:
      material(str): material.
      or_axis(str): orientation axis.
      modify_db(bool): If True database will be updated.
    """

    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE, os.path.join(material, or_axis))),
                                      gb_files, 'gb.json')
    no_struct_file = open('no_struct.txt','a')
    for gb_num, gb in enumerate(gb_files[:]):
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        json_path = '/'.join(gb[0].split('/')[7:])
        #check grain model has correct path!
        try:
            assert json_path == GB_model.path
        except AssertionError:
            print GB_model.path, json_path
            q = GrainBoundary.update(path=json_path).where(GrainBoundary.gbid==gb_json['gbid'])
            q.execute()
        #now pull subgb.json paths
        subgb_files = []
        analyze.find_gb_json('{0}'.format(gb[0]), subgb_files, 'subgb.json')
        for subgb in subgb_files:
            subgb_dict_path = os.path.join(subgb[0],'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            with open(subgb_dict_path,'r') as f:
                subgb_dict = json.load(f)

            print subgb_dict_path
            query = (GB_model.subgrains
                             .where((SubGrainBoundary.gbid == subgb_dict['name']) &
                                    (SubGrainBoundary.potential==subgb_dict['param_file'])))
            subgb_model = query.get()
            json_path = '/'.join(subgb[0].split('/')[7:])
            model_path = subgb_model.path
            try:
                assert json_path == model_path
            except AssertionError:
                #print subgb_dict['name'], subgb_model.gbid
                print json_path, model_path
                query = (SubGrainBoundary.update(path=json_path)
                                         .where((SubGrainBoundary.gbid == subgb_dict['name']) &
                                                (SubGrainBoundary.potential==subgb_dict['param_file'])))
                #print query
                query.execute()
        database.commit()
    return

def gb_check_conv(material='alphaFe', or_axis='001', modify_db=False):
    """Scans through grainboundary directory tree,
    inspecting the subgrain dictionary and the :py:class:`SubGrainBoundary` to test if
    the grain boundary energy, number of atoms, gb_area,
    and convergence flag are consistent. If modify_db is True the SQLite model
    will be updated.

    Args:
      material: Which material to do check json/database convergence consistency on.
      or_axis: Which orientation axis to check.
      modify_db: Boolean. If True updates gb_model in database otherwise
        just prints inconsistent grain json/database value.
    """

    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE, os.path.join(material, or_axis))),
                                      gb_files, 'gb.json')
    no_struct_file = open('no_struct.txt','a')
    for gb_num, gb in enumerate(gb_files[:]):
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        for subgb_model in GB_model.subgrains:
            subgb_dict_path = os.path.join(subgb_model.path,'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            with open(subgb_dict_path,'r') as f:
                subgb_dict = json.load(f)
            struct_path = os.path.join(subgb_model.path, subgb_model.gbid+'_traj.xyz')
            struct_path = os.path.join(GRAIN_DATABASE, struct_path)
            app.logger.debug(struct_path)
            try:
                assert subgb_model.converged==subgb_dict['converged']
            except AssertionError:
                if not modify_db:
                    print 'Not updating:'
                    print subgb_dict_path
                    print 'Model: ', subgb_model.converged, 'Json:', subgb_dict['converged']
                else:
                    try:
                        assert type(subgb_dict['converged'])==bool
                    except:
                        print "json 'converged' value not boolean. json file could be corrupted:"
                        print subgb_dict_path
                    else:
                        print 'Updating model instance in database:'
                        print subgb_dict_path
                        print 'Model: ', subgb_model.converged, 'Json:', subgb_dict['converged']
                        subgb_model.converged = subgb_dict['converged']
                        subgb_model.save()

            try:
                assert subgb_model.n_at==subgb_dict['n_at']
            except KeyError:
                try:
                    ats = Atoms(struct_path)
                except RuntimeError:
                    print struct_path.replace('_traj','')
                    ats = Atoms(struct_path.replace('_traj',''))

                cell = ats.get_cell()
                subgb_dict['n_at'] = len(ats)
                subgb_dict['area'] = cell[0][0]*cell[1][1]
                with open(subgb_dict_path, 'w') as f:
                    json.dump(subgb_dict, f, indent=2)
            except AssertionError:
                if not modify_db:
                    print subgb_model.n_at, subgb_dict['n_at']
                else:
                    print 'Updating model instance in database:'
                    subgb_model.n_at = subgb_dict['n_at']
                    print 'Model: {}  json:{}'.format(subgb_model.n_at, subgb_dict['n_at'])
                    subgb_model.save()

            try:
                assert (abs(subgb_model.area - subgb_dict['area']) < 1e-8)
            except KeyError:
                print 'adding area key'
                subgb_dict['area'] = subgb_dict['A']
                with open(subgb_dict_path, 'w') as f:
                    json.dump(subgb_dict, f, indent=2)
            except AssertionError:
                if not modify_db:
                    print subgb_model.area, subgb_dict['area']
                else:
                    subgb_model.area = subgb_dict['area']
                    print 'Model: {}  json:{}'.format(subgb_model.area, subgb_dict['area'])
                    subgb_model.save()

            try:
                assert (abs(subgb_model.E_gb - subgb_dict['E_gb']) < 1e-8)
            except AssertionError:
                if not modify_db:
                    print 'Not updating:'
                    print 'Model E_gb:', subgb_model.E_gb, 'JSON E_gb:',  subgb_dict['E_gb']
                else:
                    print 'Model E_gb:', subgb_model.E_gb, 'JSON E_gb:',  subgb_dict['E_gb']
                    print subgb_dict_path
                    subgb_model.E_gb = subgb_dict['E_gb']
                    subgb_model.save()
            except KeyError:
                subgb_dict['converged']=False
                subgb_dict['E_gb'] = 0.0
                with open(subgb_dict_path, 'w') as f:
                    json.dump(subgb_dict, f, indent=2)

def gb_check_force(material='alphaFe', or_axis='001', force_tol=0.05, modify_db=False, gb_start=0, sub_start=0):
    """Recurse through directory tree, loading the structure file, json dict
    and the model for each subgrain. Check that the force tolerance in the structure
    file has actually been met for convergence.

    Args:
      material(str): material string.
      or_axis(str): orientation axis.
      force_tol(float): Max force magnitude on the relaxed structures.
      modify_db(bool): If True :py:class:`SubGrainBoundary` will be updated.
      gb_start(int): Start from this grain boundary in list.
      sub_start(int): Start from this subgrainboundary in list.
    """

    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE, os.path.join(material, or_axis))),
                                                   gb_files, 'gb.json')
    no_struct_file = open('no_struct.txt','a')
    for gb_num, gb in enumerate(gb_files[gb_start:]):
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        for subgb_num, subgb_model in enumerate(GB_model.subgrains[sub_start:]):
            subgb_dict_path = os.path.join(subgb_model.path,'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            with open(subgb_dict_path,'r') as f:
                subgb_dict = json.load(f)
            struct_path = os.path.join(subgb_model.path, subgb_model.gbid+'_traj.xyz')
            struct_path = os.path.join(GRAIN_DATABASE, struct_path)
            try:
                ats = AtomsReader(struct_path)[-1]
            except RuntimeError:
                print 'No Struct File'
            except EOFError:
                print 'Struct File corrupted'
            except IOError:
                print 'No Traj File'
            else:
                print gb_num+gb_start, subgb_num+sub_start, struct_path
                try:
                    forces = [np.sqrt(x**2+y**2+z**2) for x,y,z, in zip(ats.properties['force'][0],
                                                                        ats.properties['force'][1],
                                                                        ats.properties['force'][2])]
                except KeyError:
                    print gb_num+gb_start, struct_path
                    print 'No Force in atoms object'
                    conv_check = False
                else:
                    if max(forces) <= force_tol:
                        conv_check = True
                    else:
                        conv_check = False
                        subgb_dict['E_gb'] = 0.0

            if modify_db:
                if conv_check != subgb_dict['converged']:
                    print struct_path
                    print 'Force from .xyz: ', conv_check, 'json: ', subgb_dict['converged']
                    print 'Model: ', subgb_model.converged
                    subgb_dict['converged'] = conv_check
                    with open(subgb_dict_path, 'w') as f:
                        json.dump(subgb_dict, f, indent=2)
                else:
                    pass
            else:
                try:
                    if conv_check != subgb_dict['converged']:
                        print struct_path
                        print 'Force from .xyz: ', conv_check, 'json: ', subgb_dict['converged']
                        print 'Model: ', subgb_model.converged
                    else:
                        pass
                except KeyError:
                    print 'no convergence key'
                    subgb_dict['converged']=conv_check
                    with open(subgb_dict_path, 'w') as f:
                        json.dump(subgb_dict, f, indent=2)

def change_json_key(material='alphaFe', or_axis='001'):
    """Replace '_traj' string in subgb_dict['gbid'] method.
    Can be modified to update or correct database/subgb strings in json files.

    Args:
      material(str): material to select.
      or_axis(str): orientation axis to select.
    """

    analyze  = GBAnalysis()
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE, os.path.join(material, or_axis))),
                                                   gb_files, 'gb.json')
    for gb in gb_files:
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
        for subgb_model in GB_model.subgrains:
            subgb_dict_path = os.path.join(subgb_model.path,'subgb.json')
            subgb_dict_path = os.path.join(GRAIN_DATABASE, subgb_dict_path)
            with open(subgb_dict_path,'r') as f:
                subgb_dict = json.load(f)
            try:
                assert subgb_dict['gbid'] == subgb_model.gbid
                if '_traj' in subgb_dict['gbid']:
                    print subgb_dict['gbid'], subgb_model.gbid
                    subgb_dict['gbid']= subgb_dict['gbid'].replace('_traj','')
                    with open(subgb_dict_path,'w') as f:
                        json.dump(subgb_dict, f, indent=2)
            except AssertionError:
                print subgb_dict['gbid'], subgb_model.gbid
                subgb_model.gbid = subgb_dict['gbid']
                subgb_model.save()
                print subgb_model.gbid

def insert_subgrain(material='alphaFe', or_axis='110', gbid='1108397110', json_path='/'):
    GB_model = GrainBoundary.select().where(GrainBoundary.gbid==gbid).get()

    with open(os.path.join(GRAIN_DATABASE, json_path)+'/subgb.json','r') as f:
        json_dict = json.load(f)

    model_vars = ['canonical_grain', 'converged', 'E_gb_init', 'potential',
                  'rbt', 'path', 'area', 'rcut', 'n_at', 'E_gb', 'notes', 'gbid']

    converged = json_dict['converged']
    E_gb_init = json_dict['E_gb_init']
    potential = json_dict["param_file"]
    rbt = serialize_vector(json_dict['rbt'])
    area = json_dict["rcut"]
    rcut = json_dict["area"]
    n_at = json_dict['n_at']
    E_gb = json_dict['E_gb']
    sub_gbid  = json_dict['gbid']

    subgb_dict = {"canonical_grain"  : GB_model,
                  "converged"        : converged,
                  "E_gb_init"        : E_gb_init,
                  "potential"        : potential,
                  "rbt"              : rbt,
                  "path"             : json_path,
                  "area"             : area,
                  "rcut"             : rcut,
                  "n_at"             : n_at,
                  "E_gb"             : E_gb,
                  "notes"            : "",
                  "gbid"             : sub_gbid
                }
    #try:
    print GB_model.gbid
    print subgb_dict
    print SubGrainBoundary.create(**subgb_dict)
    app.logger.info('Created entry {}'.format(subgb_dict))
    #except IntegrityError:
    #  logging.info('GB already in DB {}'.format(subgb_dict))

def populate_db(material='alphaFe', or_axis='001', gbid='', modify=False):
    """Add canonical grains to SQLite database, and all SubGrainBoundaries
    that can be found below it in the directory tree from their subgb.json files.

    Args:
      material(str): material.
      or_axis(str): orientation axis.
      gbid(str, optional): To add a specific canonical grain from its id.
      modify(bool): If True database will be updated.
    """

    analyze  = GBAnalysis()
    if len(gbid) == 0:
        dir_str  = os.path.join(material, or_axis)
    else:
        dir_str  = os.path.join(material, or_axis)
        dir_str  = os.path.join(dir_str, gbid)
    app.logger.info('dir_str {}'.format(dir_str))
    gb_files = []
    analyze.find_gb_json('{0}'.format(os.path.join(GRAIN_DATABASE, dir_str)), gb_files, 'gb.json')
    for gb in gb_files:
        app.logger.info('{} {}'.format(gb[0], gb[1]))
        with open(gb[1], 'r') as f:
            gb_json = json.load(f)
        try:
            sigma_csl = gb_json['sigma_csl']
        except KeyError:
            sigma_csl = int(gb_json['n_at']/(gb_json['coincident_sites']+gb_json['n_at']))
            gb_json['sigma_csl'] = sigma_csl
            with open(gb[1], 'w') as f:
                json.dump(gb_json, f, indent=2)

        try:
            coincident_sites = gb_json['coincident_sites']
        except KeyError:
            coincident_sites = 0

        gb_dict = {"gb_type"          : gb_json['type'],
                   "n_at"             : gb_json['n_at'],
                   "boundary_plane"   : serialize_vector(map(int, gb_json['boundary_plane'])),
                   "orientation_axis" : serialize_vector(map(int, gb_json['orientation_axis'])),
                   "z_planes"         : serialize_vector(gb_json['zplanes']),
                   "coincident_sites" : coincident_sites,
                   "sigma_csl"        : sigma_csl,
                   "angle"            : gb_json['angle'],
                   "height"           : gb_json['H'],
                   "area"             : gb_json['A'],
                   "notes"            : "",
                   "path"             : os.path.relpath(gb[0], app.config["GRAIN_DATABASE"]),
                   "gbid"             : gb_json['gbid']
                  }

        if modify:
            try:
                GB_model_object = GrainBoundary.create(**gb_dict)
            except IntegrityError:
                GB_model_object = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
                app.logger.info('GB already in database: {}'.format(gb_json['gbid']))
        else:
            try:
                GB_model_object = GrainBoundary.select().where(GrainBoundary.gbid==gb_json['gbid']).get()
                app.logger.info('Database Contains {}'.format(gb_json['gbid']))
            except  GrainBoundary.DoesNotExist:
                app.logger.info('Not in Database {}'.format(gb_json['gbid']))
                GB_model_object = None

        subgb_files = []
        analyze.find_gb_json('{0}'.format(gb[0]), subgb_files, 'subgb.json')
        for subgb in subgb_files:
            with open(subgb[1],'r') as f:
                subgb_json = json.load(f)
            try:
                converged = subgb_json['converged']
            except KeyError:
                converged = False

            try:
                E_gb = subgb_json["E_gb"]
            except KeyError:
                converged = False
                E_gb = 0.0

            try:
                E_gb_init=subgb_json["E_gb_init"]
            except KeyError:
                E_gb_init = 0.0

            try:
                gbid = subgb_json["gbid"]
            except KeyError:
                gbid = subgb_json["name"]

            try:
                area = subgb_json['A']
            except KeyError:
                structs = glob.glob(os.path.join(subgb[0], '*.xyz'))
                struct  = Atoms(structs[-1])
                cell    = struct.get_cell()
                area    = cell[0][0]*cell[1][1]
                subgb_json['n_at'] = len(struct)

            subgb_dict = {"canonical_grain" : GB_model_object,
                          "converged"       : converged,
                          "E_gb_init"       : E_gb_init,
                          "potential"       : subgb_json["param_file"],
                          "rbt"             : serialize_vector(subgb_json['rbt']),
                          "path"            : os.path.relpath(subgb[0], app.config["GRAIN_DATABASE"]),
                          "area"            : area,
                          "rcut"            : subgb_json["rcut"],
                          "n_at"            : subgb_json['n_at'],
                          "E_gb"            : E_gb,
                          "notes"           : "",
                          "gbid"            : gbid}
            if modify:
                try:
                    SubGrainBoundary.create(**subgb_dict)
                    app.logger.info('Created SubGB entry {}'.format(subgb_dict))
                except IntegrityError:
                    app.logger.info('SubGB already in DB {}'.format(subgb_dict))
            else:
                print subgb_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
#Specifiers
    parser.add_argument("-m","--material", help="The material we wish to query. Default (alphaFe)", default="alphaFe")
    parser.add_argument("-o","--or_axis", help="Orientation axis to pull from database. (default: 001)", default="001")
    parser.add_argument("-gbt","--gb_type", help="Type of grain boundary to pull: mixed, tilt, twist. (default: tilt)", default="tilt")
    parser.add_argument("-pt","--potential", help="potential to pull from database. Use --list_potential to see available potentials.", default="PotBH.xml")
    parser.add_argument("-gbid","--gbid", help="grain boundary id if present only perform action on specific boundary. applies to populate.", default="")
    parser.add_argument("-j","--json_path", help="Path (relative to Database root of subgrain to be added to database).", default="")
    parser.add_argument("-mod","--modify", help="Generic flag. If included database will be updated, otherwise program just reports intended actions  \
                                                 without  modifying database. Applies to check_conv and check_force.", action="store_true")
    parser.add_argument("-gbs", "--gb_start", help="Which grain number to start checking forces at. (default:0)", default=0, type=int)
    parser.add_argument("-sgbs", "--sub_start", help="Which grain number to start checking forces at. (default:0)", default=0, type=int)
#Actions
    parser.add_argument("-l","--list", help="List converged structures in database and their energies.", action="store_true")
    parser.add_argument("-a","--populate", help="Recurse through directory tree add subgrains to SQLite database. Choose orientation with or_axis.", action="store_true")
    parser.add_argument("-i","--insert", help="Add SubGrainboundary to database: requires gbid, or_axis, relative path to json_file.", action="store_true")
    parser.add_argument("-p","--prune", help="Remove structures from SQL database that are no longer in directory tree.", action="store_true")
    parser.add_argument("-cc","--check_conv", help="Check that convergence status of database json files corresponds to SQL database.", action="store_true")
    parser.add_argument("-cp","--check_path", help="Check that path of database json files corresponds to SQL database.", action="store_true")
    parser.add_argument("-cf","--check_force", help="Inspect xyz structure files to determine if force convergence has been reached.", action="store_true")
    parser.add_argument("-lp","--list_potential", help="List potentials in database", action="store_true")
    parser.add_argument("-la","--list_all", help="List all Grain in database along with path", action="store_true")

    args   = parser.parse_args()
    database.connect()
    if args.list:
        oraxis = ','.join([c for c in args.or_axis])
        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()

        if args.gb_type=='tilt':
            selected_grains = GrainBoundary.select().where(GrainBoundary.orientation_axis==oraxis).where(GrainBoundary.boundary_plane != oraxis)
        elif args.gb_type=='twist':
            selected_grains = GrainBoundary.select().where(GrainBoundary.orientation_axis==oraxis).where(GrainBoundary.boundary_plane == oraxis)
        else:
            sys.exit('Invalid grain boundary type.')

        print "List Minimum Energy Structures for Potential and Orientation Axis"
        print 'Material: {}, Orientation Axis: {} Potential: {}'.format(args.material, ' '.join(args.or_axis.split(',')), args.potential)
        for gb in selected_grains.order_by(GrainBoundary.angle):
            subgbs = (gb.subgrains.select(GrainBoundary, SubGrainBoundary)
                        .where(SubGrainBoundary.potential==args.potential)
                        .join(GrainBoundary).dicts())

            if len(subgbs) > 0:
                subgbs = [(16.02*(subgb['E_gb']-float(subgb['n_at']*ener_per_atom[args.potential]))/(2.0*subgb['area']), subgb) for subgb in subgbs]
                subgbs.sort(key = lambda x: x[0])
                print '\t {} {} {}'.format(round(gb.angle*(180.0/3.14159),3), round(subgbs[0][0],3), subgbs[0][1]['path'])

    if args.list_all:
        for gb in GrainBoundary.select().order_by(GrainBoundary.orientation_axis).order_by(GrainBoundary.angle):
            print gb.orientation_axis, gb.angle*(180/np.pi), gb.gb_type, gb.path

        for gb in GrainBoundary.select().order_by(GrainBoundary.orientation_axis).order_by(GrainBoundary.angle):
            for subgb in gb.subgrains:
                print gb.orientation_axis, gb.angle*(180/np.pi), subgb.potential, subgb.path

    if args.list_potential:
        pot_param     = PotentialParameters()
        ener_per_atom = pot_param.gs_ener_per_atom()
        for k, v in ener_per_atom.items():
            print 'Potential Name: {}, Per Atom Energy: {}'.format(k, v)

    if args.populate:
        populate_db(material=args.material, or_axis=args.or_axis, modify=args.modify, gbid=args.gbid)

    if args.prune:
        gb_check_dir_integrity(material=args.material, or_axis=args.or_axis)

    if args.check_conv:
        gb_check_conv(material=args.material, or_axis=args.or_axis, modify_db=args.modify)

    if args.check_force:
        gb_check_force(material=args.material, or_axis=args.or_axis, modify_db=args.modify, gb_start=args.gb_start, sub_start=args.sub_start,
                       gbid=args.gbid)

    if args.insert:
        assert args.gbid != ''
        assert args.json_path != ''
        insert_subgrain(material=args.material, or_axis=args.or_axis, gbid=args.gbid, json_path=args.json_path)

    if args.check_path:
        gb_check_path(material=args.material, or_axis=args.or_axis, modify_db=args.modify)
    database.close()
